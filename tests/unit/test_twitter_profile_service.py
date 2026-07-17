from unittest.mock import patch

from src.extraction.services.twitter_profile import (
    TwitterApiError,
    TwitterProfileService,
    tw_username,
)


def test_handle_normalization_rejects_path_and_keeps_valid_handle() -> None:
    assert tw_username("https://x.com/News_Arab/status/123") == "News_Arab"
    assert tw_username("@news_arab") == "news_arab"
    assert TwitterProfileService()._do_fetch("not a valid handle!").exists is False


def test_recommendations_clamp_limit_and_normalize_optional_fields() -> None:
    rows = [
        {
            "user": {
                "screen_name": "ArabNews",
                "name": "Arab News",
                "followers_count": 12,
                "description": " أخبار ",
                "profile_image_url_https": "https://img.test/avatar_normal.jpg",
                "id_str": "42",
            }
        },
        {"user": {}},
    ]
    with patch(
        "src.extraction.services.twitter_profile._rest_recommendations",
        return_value=rows,
    ) as recommendations:
        result = TwitterProfileService()._do_recommendations("@seed", 999)

    assert recommendations.call_args.args[1] == 40
    assert result.exists is True
    assert len(result.recommendations) == 1
    account = result.recommendations[0]
    assert account.username == "arabnews"
    assert account.description == "أخبار"
    assert account.image_url == "https://img.test/avatar_400x400.jpg"


def test_recommendations_report_rate_limit_without_upstream_detail() -> None:
    with patch(
        "src.extraction.services.twitter_profile._rest_recommendations",
        side_effect=TwitterApiError(429, rate_limited=True),
    ):
        result = TwitterProfileService()._do_recommendations("seed", 10)

    assert result.exists is False
    assert result.rate_limited is True
    assert result.recommendations == []


def test_profile_parser_preserves_post_and_interaction_graph_contracts() -> None:
    author = {
        "screen_name": "News_Arab",
        "name": "Arabic News",
        "followers_count": 42,
        "verified": True,
        "description": "أخبار",
        "profile_image_url_https": "https://img.test/avatar_normal.jpg",
    }
    tweet = {
        "legacy": {
            "id_str": "123",
            "full_text": "خبر @Mentioned",
            "created_at": "Tue Jul 14 12:00:00 +0000 2026",
            "favorite_count": 7,
            "retweet_count": 3,
            "reply_count": 2,
            "entities": {"user_mentions": [{"screen_name": "Mentioned"}]},
            "retweeted_status_result": {
                "result": {
                    "core": {
                        "user_results": {"result": {"legacy": {"screen_name": "Retweeted"}}}
                    }
                }
            },
            "quoted_status_result": {
                "result": {
                    "core": {
                        "user_results": {"result": {"legacy": {"screen_name": "Quoted"}}}
                    }
                }
            },
        },
        "core": {"user_results": {"result": {"legacy": author}}},
    }
    data = {
        "data": {
            "user": {
                "result": {
                    "timeline": {
                        "instructions": [
                            {
                                "type": "TimelineAddEntries",
                                "entries": [
                                    {
                                        "entryId": "tweet-123",
                                        "content": {
                                            "itemContent": {"tweet_results": {"result": tweet}}
                                        },
                                    }
                                ],
                            }
                        ]
                    }
                }
            }
        }
    }

    result = TwitterProfileService()._parse("news_arab", data)

    assert result.exists is True
    assert result.name == "Arabic News"
    assert result.image_url == "https://img.test/avatar_400x400.jpg"
    assert result.posts[0].model_dump() == {
        "id": "123",
        "text": "خبر @Mentioned",
        "created_at": "2026-07-14T12:00:00+00:00",
        "url": "https://x.com/news_arab/status/123",
        "likes": 7,
        "retweets": 3,
        "replies": 2,
        "is_retweet": True,
        "is_reply": False,
    }
    assert result.retweeted == ["retweeted"]
    assert result.quoted == ["quoted"]
    assert result.mentioned == ["mentioned"]
