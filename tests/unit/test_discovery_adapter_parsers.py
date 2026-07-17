import json
from types import SimpleNamespace
from unittest.mock import patch

from src.extraction.services.apple_podcast import ApplePodcastService
from src.extraction.services.telegram_channel import (
    _channel_from_href,
    _parse_subscribers,
    tg_username,
)
from src.extraction.services.youtube_innertube import YouTubeInnerTubeService


class FixtureNode:
    """Minimal Scrapling-compatible node used only for sanitized parser shapes."""

    def __init__(
        self,
        text: str = "",
        *,
        attrib: dict[str, str] | None = None,
        children: dict[str, list["FixtureNode"]] | None = None,
    ) -> None:
        self._text = text
        self.attrib = attrib or {}
        self._children = children or {}

    def css(self, selector: str) -> list["FixtureNode"]:
        return self._children.get(selector, [])

    def get_all_text(self) -> str:
        return self._text


def test_telegram_normalization_and_relation_filtering() -> None:
    assert tg_username("https://t.me/s/News_Arab?x=1") == "news_arab"
    assert _parse_subscribers("1.5M subscribers") == 1_500_000
    assert _channel_from_href("https://t.me/OtherNews", "news_arab") == "othernews"
    assert _channel_from_href("https://t.me/news_arab", "news_arab") is None
    assert _channel_from_href("https://t.me/alertbot", "news_arab") is None


def test_telegram_parser_normalizes_posts_and_graph_edges() -> None:
    text = FixtureNode(
        "خبر عربي",
        children={
            "a": [
                FixtureNode(attrib={"href": "https://t.me/OtherNews"}),
                FixtureNode(attrib={"href": "https://t.me/news_arab"}),
            ]
        },
    )
    message = FixtureNode(
        children={
            ".tgme_widget_message_text": [text],
            ".tgme_widget_message_date time": [
                FixtureNode(attrib={"datetime": "2026-07-14T12:00:00+00:00"})
            ],
            ".tgme_widget_message_views": [FixtureNode("120")],
            "a.tgme_widget_message_forwarded_from_name": [
                FixtureNode(attrib={"href": "https://t.me/ForwardedNews"})
            ],
        }
    )
    counter = FixtureNode(
        children={
            ".counter_type": [FixtureNode("Subscribers")],
            ".counter_value": [FixtureNode("1.5K")],
        }
    )
    page = FixtureNode(
        children={
            ".tgme_widget_message": [message],
            ".tgme_channel_info_header_title": [FixtureNode("Arabic News")],
            ".tgme_channel_info_counter": [counter],
        }
    )
    fetcher = SimpleNamespace(get=lambda *_args, **_kwargs: page)
    with patch(
        "src.extraction.services.telegram_channel._get_fetcher", return_value=fetcher
    ):
        from src.extraction.services.telegram_channel import TelegramChannelService

        result = TelegramChannelService()._do_fetch("https://t.me/s/News_Arab")

    assert result.username == "news_arab"
    assert result.exists is True
    assert result.subscribers == 1_500
    assert result.posts[0].model_dump() == {
        "text": "خبر عربي",
        "datetime": "2026-07-14T12:00:00+00:00",
        "views": "120",
    }
    assert result.forwarded == ["forwardednews"]
    assert result.mentioned == ["othernews"]


def test_apple_related_parser_deduplicates_self_and_caps_results() -> None:
    items = [
        {"adamId": "100", "title": "Self", "releaseFrequency": "weekly"},
        {"adamId": "200", "title": "Arabic show", "genreNames": ["News"]},
        {"adamId": "200", "title": "Duplicate"},
        *[
            {"adamId": str(300 + index), "title": f"Show {index}"}
            for index in range(30)
        ],
    ]
    html = (
        '<script id="serialized-server-data">'
        '{"section":{"title":"You Might Also Like","items":'
        + json.dumps(items)
        + "}}</script>"
    )
    fetcher = SimpleNamespace(
        get=lambda *_args, **_kwargs: SimpleNamespace(body=html.encode())
    )
    with patch(
        "src.extraction.services.apple_podcast._get_fetcher", return_value=fetcher
    ):
        result = ApplePodcastService()._do_fetch_related("100", "US")

    assert result.exists is True
    assert result.collection_id == "100"
    assert [show.adam_id for show in result.related].count("200") == 1
    assert all(show.adam_id != "100" for show in result.related)
    assert len(result.related) == 25


def test_apple_malformed_payload_has_stable_empty_result() -> None:
    fetcher = SimpleNamespace(
        get=lambda *_args, **_kwargs: SimpleNamespace(
            body=b'<script id="serialized-server-data">not-json</script>'
        )
    )
    with patch(
        "src.extraction.services.apple_podcast._get_fetcher", return_value=fetcher
    ):
        result = ApplePodcastService()._do_fetch_related("123", "us")
    assert result.exists is False
    assert result.related == []


def test_youtube_feed_parser_deduplicates_channel_mentions() -> None:
    channel_id = "UC" + "a" * 22
    raw = {
        "contents": [
            {"channelRenderer": {"channelId": channel_id, "title": {"simpleText": "News"}}},
            {"channelRenderer": {"channelId": channel_id, "title": {"simpleText": "Duplicate"}}},
        ]
    }
    channels = YouTubeInnerTubeService()._do_parse_feed(raw)
    assert len(channels) == 1
    assert channels[0].channel_id == channel_id
    assert channels[0].mention_count == 2


def test_youtube_raw_channel_id_resolves_without_transport() -> None:
    channel_id = "UC" + "b" * 22
    with patch("src.extraction.services.youtube_innertube._post") as post:
        channels = YouTubeInnerTubeService()._do_resolve_links([channel_id, channel_id])
    assert [channel.channel_id for channel in channels] == [channel_id]
    post.assert_not_called()


def test_youtube_search_normalizes_deduplicates_and_honors_limit() -> None:
    first = "UC" + "c" * 22
    second = "UC" + "d" * 22
    payload = {
        "items": [
            {
                "channelRenderer": {
                    "channelId": first,
                    "title": {"simpleText": "أخبار عربية"},
                    "videoCountText": {"simpleText": "1.2M subscribers"},
                }
            },
            {"channelRenderer": {"channelId": first}},
            {
                "channelRenderer": {
                    "channelId": second,
                    "title": {"simpleText": "English news"},
                }
            },
        ]
    }
    with patch(
        "src.extraction.services.youtube_innertube._post", return_value=payload
    ) as post:
        result = YouTubeInnerTubeService()._do_search_channels(" news ", 1)

    assert result.query == "news"
    assert [(channel.channel_id, channel.subscribers) for channel in result.channels] == [
        (first, 1_200_000)
    ]
    assert post.call_args.args[0] == "search"


def test_youtube_transport_failure_returns_a_stable_empty_result() -> None:
    with patch(
        "src.extraction.services.youtube_innertube._post", side_effect=TimeoutError("upstream")
    ):
        result = YouTubeInnerTubeService()._do_search_channels("news", 10)

    assert result.query == "news"
    assert result.channels == []


def test_youtube_handle_resolution_uses_fake_innertube_response() -> None:
    channel_id = "UC" + "e" * 22
    payload = {
        "endpoint": {
            "browseEndpoint": {
                "browseId": channel_id,
                "canonicalBaseUrl": "/@arab-news",
            }
        }
    }
    with patch("src.extraction.services.youtube_innertube._post", return_value=payload):
        channels = YouTubeInnerTubeService()._do_resolve_links(["@arab-news"])

    assert [(channel.channel_id, channel.handle) for channel in channels] == [
        (channel_id, "arab-news")
    ]
