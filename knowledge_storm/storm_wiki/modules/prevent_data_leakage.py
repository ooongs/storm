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
    "eaglepubs.erau.edu",
}

LEAKAGE_HOST_KEYWORDS = {
    "pressbooks",
}


def _normalize_host(url_or_host):
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


def is_leakage_url(url):
    host = _normalize_host(url)
    if not host:
        return False

    for domain in LEAKAGE_DOMAIN_PATTERNS:
        normalized_domain = _normalize_host(domain)
        if host == normalized_domain or host.endswith(f".{normalized_domain}"):
            return True

    return any(keyword in host for keyword in LEAKAGE_HOST_KEYWORDS)


def is_allowed_source(url):
    return not is_leakage_url(url)
