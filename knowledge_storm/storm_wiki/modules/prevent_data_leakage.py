LEAKAGE_DOMAIN_PATTERNS = {
    "cnx.org",
    "openstax.org",
    "pressbooks.pub",
    "pb.unizin.org",
    "libretexts.org",
    "courses.lumenlearning.com",
    "lumenlearning.com",
    "open.lib.umn.edu",
    "open.oregonstate.education",
    "open.maricopa.edu",
    "openwa.pressbooks.pub",
    "openfl.pressbooks.pub",
    "openoregon.pressbooks.pub",
    "louis.pressbooks.pub",
    "lmu.pressbooks.pub",
    "minnstate.pressbooks.pub",
    "ecampusontario.pressbooks.pub",
    "pressbooks.atlanticoer-relatlantique.ca",
    "erau.edu",
    "eaglepubs.erau.edu",
    "commons.erau.edu",
}

LEAKAGE_HOST_KEYWORDS = {
    "pressbooks",
}


def _normalize_host(url_or_host):
    url_or_host = _source_url(url_or_host)
    value = (url_or_host or "").strip().lower()
    if not value:
        return ""
    if "://" not in value:
        value = f"https://{value}"

    try:
        from urllib.parse import urlparse

        host = urlparse(value).netloc or urlparse(value).path
    except Exception:
        host = value

    host = host.split("@")[-1].split(":")[0].strip(".")
    if host.startswith("www."):
        host = host[4:]
    return host


def _source_url(source):
    if isinstance(source, dict):
        return (
            source.get("url")
            or source.get("link")
            or source.get("host")
            or ""
        )
    return source or ""


def _source_text(source):
    if isinstance(source, dict):
        parts = [
            source.get("url", ""),
            source.get("link", ""),
            source.get("title", ""),
            source.get("description", ""),
        ]
        parts.extend(source.get("snippets") or [])
        return " ".join(str(part or "") for part in parts).lower()
    return str(source or "").lower()


def _url_path(source):
    value = (_source_url(source) or "").strip()
    if not value:
        return ""
    if "://" not in value:
        value = f"https://{value}"
    try:
        from urllib.parse import urlparse

        return urlparse(value).path.lower()
    except Exception:
        return value.lower()


def _is_edu_host(host):
    return host == "edu" or host.endswith(".edu")


def _is_edu_pdf_source(source, host):
    if not _is_edu_host(host):
        return False
    path = _url_path(source)
    text = _source_text(source)
    return path.endswith(".pdf") or ".pdf/" in path or "[pdf]" in text


def is_leakage_url(source):
    host = _normalize_host(source)
    if not host:
        return False

    if _is_edu_pdf_source(source, host):
        return True

    for domain in LEAKAGE_DOMAIN_PATTERNS:
        normalized_domain = _normalize_host(domain)
        if host == normalized_domain or host.endswith(f".{normalized_domain}"):
            return True

    return any(keyword in host for keyword in LEAKAGE_HOST_KEYWORDS)


def is_allowed_source(url):
    return not is_leakage_url(url)
