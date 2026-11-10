"""Route table guards: no path+method may be registered twice."""
from unified_server import app


def test_no_duplicate_method_path_pairs():
    seen = {}
    duplicates = []
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None) or set()
        if not path:
            continue
        for method in methods:
            key = (method, path)
            if key in seen:
                duplicates.append((method, path, seen[key], getattr(route, "name", "")))
            seen[key] = getattr(route, "name", "")
    assert duplicates == [], f"duplicate routes: {duplicates}"


def test_ocr_route_is_registered_once_and_works():
    ocr = [r for r in app.routes
           if getattr(r, "path", None) == "/v1/images/ocr"
           and "POST" in (getattr(r, "methods", None) or set())]
    assert len(ocr) == 1
    assert ocr[0].endpoint.__name__ == "ocr_endpoint"
