def test_declared_oversized_body_is_rejected_before_route_work(client, auth_headers) -> None:
    response = client.post(
        "/v1/embed",
        content=b"{}",
        headers={
            **auth_headers,
            "Content-Type": "application/json",
            "Content-Length": "1000001",
        },
    )
    assert response.status_code == 413
    assert response.json()["error_code"] == "REQUEST_BODY_TOO_LARGE"
