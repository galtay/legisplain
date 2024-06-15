from bs4 import BeautifulSoup
from unstructured.cleaners.core import clean


def get_legis_text_v1(xml: str):

    soup = BeautifulSoup(xml, "xml")
    main_keys = [ch.name for ch in soup.children if ch.name]
    assert len(main_keys) == 1
    main_key = main_keys[0]
    name_text_children = [
        (
            child.name,
            clean(
                child.get_text(separator=" ").strip().replace("\t", " "),
                extra_whitespace=True,
            ),
        )
        for child in soup.find(main_key)
        if (
            child.name != "metadata"
            and child.name is not None
            and child.get_text().strip() != ""
        )
    ]
    text = "\n\n".join([f"{name_text[1]}" for name_text in name_text_children])
    return text
