import http.client, urllib.parse, gzip, random


class QuickLatex(object):
    host = "quicklatex.com"
    
    path = "/latex3.f"
    
    headers = { "Host": "quicklatex.com",
        "Connection": "keep-alive",
        "Accept": "*/*",
        "DNT": "1",
        "X-Requested-With": "XMLHttpRequest",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36",
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://quicklatex.com",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Dest": "empty",
        "Referer": "https://quicklatex.com/",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-GB,en;q=0.9,de-DE;q=0.8,de;q=0.7,en-US;q=0.6" }



def render(latex_snippet):
    """Send latex_snippet to quicklatex.com for rendering, and return the resulting image URL.
    """
    BODY=latex_snippet
    PREAMBLE='\n'.join(r"""
    \usepackage{booktabs}
    \usepackage[dvipsnames]{xcolor}
    """.strip().split())

    formdata = f"formula={BODY.replace('%', '%25').replace('&', '%26')}&fsize=17px&fcolor=000000&bcolor=ffffff&mode=0&out=1&remhost=quicklatex.com&preamble={PREAMBLE}&rnd={random.random()*100}"
    conn = http.client.HTTPSConnection(QuickLatex.host)
    conn.request("POST", QuickLatex.path, formdata, dict(QuickLatex.headers))
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError(f'{response.status}: {response.reason}; {response.read()}')
    data = response.read()
    conn.close()
    data = gzip.decompress(data).decode('utf8')
    if 'error' in data:
        raise RuntimeError(data)
    return data.split()[1]


def display(latex_snippet):
    """Send latex_snippet to quicklatex.com for rendering, and return an IPython Image for display in a notebook.
    """
    from IPython.display import Image
    return Image(url=render(latex_snippet))