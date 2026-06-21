"""X/Twitter public-profile reader (guest-token GraphQL).

X soft-bans the syndication `timeline-profile` endpoint by IP under load. Instead
we use the same path snscrape-style tools use: activate a **guest token** (no
login) against the public web bearer, then call the GraphQL `UserByScreenName` +
`UserTweets` endpoints. This runs on `api.twitter.com` (not the banned syndication
host), needs no API key, and returns the full recent timeline as JSON. Requests go
through curl_cffi browser-impersonation. Serves both the Source Intelligence
interaction-graph (retweet/quote/mention edges + followers) and ingestion (full
posts → TWEET content items).

GraphQL queryIds rotate when X ships a new web build; override via env
(`TWITTER_GQL_USER_BY_SCREENNAME`, `TWITTER_GQL_USER_TWEETS`) without a code change.
"""

import asyncio
import json
import os
import re
import time
from datetime import datetime, timezone

from curl_cffi import requests as cffi

from src.common.utils.logging import get_logger
from src.extraction.schemas.extract import (
    TwitterPost,
    TwitterProfileResponse,
    TwitterRecAccount,
    TwitterRecommendationsResponse,
)

logger = get_logger(__name__)

_HANDLE_RE = re.compile(r"^[A-Za-z0-9_]{1,15}$")

# Public web-app bearer (embedded in twitter.com JS — not a secret). Env-overridable.
_WEB_BEARER = os.getenv(
    "TWITTER_WEB_BEARER",
    "AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs"
    "%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA",
)
_QID_USER = os.getenv("TWITTER_GQL_USER_BY_SCREENNAME", "G3KGOASz96M-Qu0nwmGXNg")
_QID_TWEETS = os.getenv("TWITTER_GQL_USER_TWEETS", "E3opETHurmVJflFsUBVuUQ")
_GQL = "https://api.twitter.com/graphql/{qid}/{op}"

_USER_FEATURES = {
    "hidden_profile_subscriptions_enabled": True,
    "rweb_tipjar_consumption_enabled": True,
    "responsive_web_graphql_exclude_directive_enabled": True,
    "verified_phone_label_enabled": False,
    "subscriptions_verification_info_is_identity_verified_enabled": True,
    "subscriptions_verification_info_verified_since_enabled": True,
    "highlights_tweets_tab_ui_enabled": True,
    "responsive_web_twitter_article_notes_tab_enabled": True,
    "subscriptions_feature_can_gift_premium": True,
    "creator_subscriptions_tweet_preview_api_enabled": True,
    "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
    "responsive_web_graphql_timeline_navigation_enabled": True,
}
_TWEET_FEATURES = {
    "rweb_tipjar_consumption_enabled": True,
    "responsive_web_graphql_exclude_directive_enabled": True,
    "verified_phone_label_enabled": False,
    "creator_subscriptions_tweet_preview_api_enabled": True,
    "responsive_web_graphql_timeline_navigation_enabled": True,
    "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
    "communities_web_enable_tweet_community_results_fetch": True,
    "c9s_tweet_anatomy_moderator_badge_enabled": True,
    "articles_preview_enabled": True,
    "responsive_web_edit_tweet_api_enabled": True,
    "graphql_is_translatable_rweb_tweet_is_translatable_enabled": True,
    "view_counts_everywhere_api_enabled": True,
    "longform_notetweets_consumption_enabled": True,
    "responsive_web_twitter_article_tweet_consumption_enabled": True,
    "tweet_awards_web_tipping_enabled": False,
    "creator_subscriptions_quote_tweet_preview_enabled": False,
    "freedom_of_speech_not_reach_fetch_enabled": True,
    "standardized_nudges_misinfo": True,
    "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": True,
    "rweb_video_timestamps_enabled": True,
    "longform_notetweets_rich_text_read_enabled": True,
    "longform_notetweets_inline_media_enabled": True,
    "responsive_web_enhance_cards_enabled": False,
}

_GUEST_TTL = 3 * 3600
_guest = {"token": None, "ts": 0.0}
_id_cache: dict[str, str] = {}


def tw_username(raw: str) -> str:
    """Normalize a bare handle, @handle, or x.com/twitter.com URL to a handle."""
    s = (raw or "").strip()
    s = re.sub(r"^https?://", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(www\.)?(x|twitter|mobile\.twitter)\.com/", "", s, flags=re.IGNORECASE)
    s = s.lstrip("@")
    s = re.split(r"[/?#]", s, maxsplit=1)[0]
    return s.strip()


def _iso(created_at: str | None) -> str | None:
    if not created_at:
        return None
    try:
        dt = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
        return dt.astimezone(timezone.utc).isoformat()
    except ValueError:
        return created_at


def _headers(guest_token: str) -> dict:
    return {
        "Authorization": f"Bearer {_WEB_BEARER}",
        "x-guest-token": guest_token,
        "Content-Type": "application/json",
    }


def _guest_token(force: bool = False) -> str:
    if not force and _guest["token"] and (time.time() - _guest["ts"]) < _GUEST_TTL:
        return _guest["token"]
    r = cffi.post(
        "https://api.twitter.com/1.1/guest/activate.json",
        headers={"Authorization": f"Bearer {_WEB_BEARER}"},
        impersonate="chrome",
        timeout=20,
    )
    r.raise_for_status()
    token = r.json()["guest_token"]
    _guest.update(token=token, ts=time.time())
    return token


class TwitterApiError(Exception):
    """Raised when a GraphQL call fails after a guest-token refresh.

    Carries whether the failure was a throttle so callers can surface an accurate
    rate-limited signal (vs. brittle substring-sniffing of the message).
    """

    def __init__(self, status: int, rate_limited: bool):
        self.status = status
        self.rate_limited = rate_limited
        super().__init__(f"twitter graphql failed: status={status} rate_limited={rate_limited}")


def _gql(qid: str, op: str, variables: dict, features: dict, *, retry: bool = True) -> dict:
    token = _guest_token()
    r = cffi.get(
        _GQL.format(qid=qid, op=op),
        params={"variables": json.dumps(variables), "features": json.dumps(features)},
        headers=_headers(token),
        impersonate="chrome",
        timeout=25,
    )
    if r.status_code in (401, 403, 429):
        # Guest token expired / per-token quota hit — refresh once and retry.
        if retry:
            _guest_token(force=True)
            return _gql(qid, op, variables, features, retry=False)
        raise TwitterApiError(r.status_code, rate_limited=r.status_code == 429)
    r.raise_for_status()
    data = r.json()
    # X often returns HTTP 200 with a GraphQL `errors` array (suspended/protected
    # users, or a soft rate-limit). Retry once on a throttle; otherwise let the
    # caller treat a missing `user.result` as not-found.
    errs = data.get("errors")
    if errs:
        msg = " ".join(str(e.get("message", "")) for e in errs).lower()
        throttled = "rate" in msg or "limit" in msg or "over capacity" in msg
        if throttled and retry:
            _guest_token(force=True)
            return _gql(qid, op, variables, features, retry=False)
        if throttled:
            raise TwitterApiError(200, rate_limited=True)
    return data


def _resolve_user_id(handle: str) -> str | None:
    key = handle.lower()
    if key in _id_cache:
        return _id_cache[key]
    data = _gql(_QID_USER, "UserByScreenName", {"screen_name": handle}, _USER_FEATURES)
    result = (((data.get("data") or {}).get("user") or {}).get("result")) or {}
    uid = result.get("rest_id")
    if uid:
        if len(_id_cache) > 5000:  # bound the process-global cache
            _id_cache.clear()
        _id_cache[key] = uid
    return uid


def _unwrap(result: dict | None) -> dict | None:
    """A tweet result is either a Tweet or a TweetWithVisibilityResults wrapper."""
    if not result:
        return None
    if result.get("__typename") == "TweetWithVisibilityResults":
        return result.get("tweet")
    return result


def _author(tweet: dict) -> dict:
    return ((tweet.get("core") or {}).get("user_results") or {}).get("result", {}).get("legacy", {})


def _expanded_url(user: dict) -> str | None:
    """The account's bio website (entities.url.urls[0].expanded_url), if any."""
    urls = (((user.get("entities") or {}).get("url") or {}).get("urls")) or []
    return urls[0].get("expanded_url") if urls else None


def _avatar(user: dict) -> str | None:
    """Profile image, upsized from the default 48px `_normal` to `_400x400`.
    Free — present inline on every user object; served from the public pbs CDN.
    """
    u = user.get("profile_image_url_https") or user.get("profile_image_url")
    return u.replace("_normal", "_400x400") if u else None


def _rest_recommendations(seed_param: dict, limit: int, *, retry: bool = True) -> list[dict]:
    """GET users/recommendations.json with a guest token (the connect_people
    backend). `display_location=profile_accounts_sidebar` is REQUIRED — every
    other value 500s. Returns the raw recommendation list (each {user, user_id}).
    """
    token = _guest_token()
    params = {
        **seed_param,
        "connections": "following",
        "limit": limit,
        "display_location": "profile_accounts_sidebar",
        "pc": "true",
    }
    headers = {**_headers(token), "x-twitter-active-user": "yes"}
    r = cffi.get(
        "https://api.twitter.com/1.1/users/recommendations.json",
        params=params,
        headers=headers,
        impersonate="chrome",
        timeout=25,
    )
    if r.status_code in (401, 403, 429):
        # Guest token expired / per-token quota — refresh once and retry.
        if retry:
            _guest_token(force=True)
            return _rest_recommendations(seed_param, limit, retry=False)
        raise TwitterApiError(r.status_code, rate_limited=r.status_code == 429)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


class TwitterProfileService:
    """Fetch + parse one X profile timeline via guest-token GraphQL."""

    def __init__(self, timeout_sec: int = 30):
        self.timeout_sec = timeout_sec

    async def fetch(self, username: str) -> TwitterProfileResponse:
        return await asyncio.to_thread(self._do_fetch, username)

    async def fetch_recommendations(
        self, seed: str, limit: int = 40
    ) -> TwitterRecommendationsResponse:
        return await asyncio.to_thread(self._do_recommendations, seed, limit)

    def _do_recommendations(
        self, raw_seed: str, limit: int
    ) -> TwitterRecommendationsResponse:
        raw = (raw_seed or "").strip()
        # Numeric → user_id; otherwise normalize a handle/@handle/URL → screen_name.
        if raw.isdigit():
            seed_param: dict = {"user_id": raw}
        else:
            handle = tw_username(raw)
            if not _HANDLE_RE.match(handle):
                return TwitterRecommendationsResponse(seed=raw, exists=False)
            seed_param = {"screen_name": handle}

        try:
            rows = _rest_recommendations(seed_param, max(1, min(limit, 40)))
        except TwitterApiError as exc:
            logger.warning("twitter_recs_failed", seed=raw, status=exc.status)
            return TwitterRecommendationsResponse(
                seed=raw, exists=False, rate_limited=exc.rate_limited
            )
        except Exception as exc:  # noqa: BLE001
            rl = "429" in str(exc)
            logger.warning("twitter_recs_failed", seed=raw, error=str(exc))
            return TwitterRecommendationsResponse(seed=raw, exists=False, rate_limited=rl)

        recs: list[TwitterRecAccount] = []
        for row in rows:
            u = row.get("user") or {}
            sn = u.get("screen_name")
            if not sn:
                continue
            recs.append(
                TwitterRecAccount(
                    username=sn.lower(),
                    name=u.get("name"),
                    followers=int(u.get("followers_count") or 0),
                    friends=int(u.get("friends_count") or 0),
                    statuses=int(u.get("statuses_count") or 0),
                    listed=int(u.get("listed_count") or 0),
                    verified=bool(u.get("verified")) or bool(u.get("ext_is_blue_verified")),
                    is_protected=bool(u.get("protected")),
                    description=(u.get("description") or "").strip(),
                    url=_expanded_url(u),
                    image_url=_avatar(u),
                    created_at=_iso(u.get("created_at")),
                    user_id=str(u.get("id_str") or u.get("id") or "") or None,
                )
            )
        return TwitterRecommendationsResponse(seed=raw, exists=True, recommendations=recs)

    def _do_fetch(self, raw_username: str) -> TwitterProfileResponse:
        user = tw_username(raw_username)
        if not _HANDLE_RE.match(user):
            return TwitterProfileResponse(username=user, exists=False)

        try:
            uid = _resolve_user_id(user)
        except TwitterApiError as exc:
            logger.warning("twitter_resolve_failed", username=user, status=exc.status)
            return TwitterProfileResponse(
                username=user, exists=False, rate_limited=exc.rate_limited
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("twitter_resolve_failed", username=user, error=str(exc))
            return TwitterProfileResponse(username=user, exists=False)
        if not uid:
            return TwitterProfileResponse(username=user, exists=False)

        try:
            data = _gql(
                _QID_TWEETS,
                "UserTweets",
                {"userId": uid, "count": 20, "includePromotedContent": False, "withVoice": False},
                _TWEET_FEATURES,
            )
        except TwitterApiError as exc:
            logger.warning("twitter_tweets_failed", username=user, status=exc.status)
            return TwitterProfileResponse(
                username=user, exists=False, rate_limited=exc.rate_limited
            )
        except Exception as exc:  # noqa: BLE001
            rl = "429" in str(exc)
            logger.warning("twitter_tweets_failed", username=user, error=str(exc))
            return TwitterProfileResponse(username=user, exists=False, rate_limited=rl)

        return self._parse(user, data)

    def _parse(self, user: str, data: dict) -> TwitterProfileResponse:
        self_lower = user.lower()
        posts: list[TwitterPost] = []
        retweeted: set[str] = set()
        quoted: set[str] = set()
        mentioned: set[str] = set()
        name: str | None = None
        followers = 0
        verified = False
        image_url: str | None = None

        def edge(sn: str | None, bucket: set[str]) -> None:
            if sn and sn.lower() != self_lower:
                bucket.add(sn.lower())

        # Walk timeline instructions → entries → tweet results. Depending on the
        # queryId the timeline sits at result.timeline OR result.timeline_v2.timeline.
        result = ((data.get("data") or {}).get("user") or {}).get("result") or {}
        tl = result.get("timeline_v2") or result.get("timeline") or {}
        tl = tl.get("timeline", tl)  # unwrap the extra level if present
        instructions = tl.get("instructions", [])
        entries = []
        for ins in instructions:
            if ins.get("type") == "TimelineAddEntries":
                entries = ins.get("entries", [])
                break

        for entry in entries:
            if not str(entry.get("entryId", "")).startswith("tweet-"):
                continue
            result = _unwrap(
                (
                    ((entry.get("content") or {}).get("itemContent") or {}).get("tweet_results")
                    or {}
                ).get("result")
            )
            if not result or "legacy" not in result:
                continue
            lg = result["legacy"]
            author = _author(result)
            if author.get("screen_name", "").lower() == self_lower:
                name = name or author.get("name")
                followers = followers or int(author.get("followers_count") or 0)
                verified = (
                    verified or bool(author.get("verified")) or bool(result.get("is_blue_verified"))
                )
                image_url = image_url or _avatar(author)

            rt = _unwrap((lg.get("retweeted_status_result") or {}).get("result"))
            qt = _unwrap((lg.get("quoted_status_result") or {}).get("result"))
            if rt:
                edge(_author(rt).get("screen_name"), retweeted)
            if qt:
                edge(_author(qt).get("screen_name"), quoted)
            for men in (lg.get("entities") or {}).get("user_mentions") or []:
                edge(men.get("screen_name"), mentioned)

            tid = str(lg.get("id_str") or "")
            posts.append(
                TwitterPost(
                    id=tid,
                    text=(lg.get("full_text") or "").strip(),
                    created_at=_iso(lg.get("created_at")),
                    url=f"https://x.com/{user}/status/{tid}" if tid else None,
                    likes=int(lg.get("favorite_count") or 0),
                    retweets=int(lg.get("retweet_count") or 0),
                    replies=int(lg.get("reply_count") or 0),
                    is_retweet=bool(rt),
                    is_reply=bool(lg.get("in_reply_to_status_id_str")),
                )
            )

        if not posts:
            return TwitterProfileResponse(username=user, exists=False)

        return TwitterProfileResponse(
            username=user,
            exists=True,
            name=name,
            followers=followers,
            verified=verified,
            image_url=image_url,
            posts=posts,
            retweeted=sorted(retweeted),
            quoted=sorted(quoted),
            mentioned=sorted(mentioned),
        )
