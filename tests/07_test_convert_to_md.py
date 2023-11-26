import json

from tqdm import tqdm
from datetime import datetime
from utils.db import get_papers_db

def render_paper(paper_entry: dict, idx: int) -> str:
    # get the arxiv id
    arxiv_id = paper_entry["arxiv_id"]
    # get the title
    title = paper_entry["title"]
    # get the arxiv url
    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
    # get the abstract
    abstract = paper_entry["abstract"]
    # get the authors
    authors = paper_entry["authors"]
    paper_string = f'## {idx}. [{title}]({arxiv_url}) <a id="link{idx}"></a>\n'
    paper_string += f"**ArXiv ID:** {arxiv_id}\n"
    paper_string += f'**Authors:** {", ".join(authors)}\n\n'
    paper_string += f"**Abstract:** {abstract}\n\n"

    return paper_string + "\n---\n"


def render_title_and_author(paper_entry: dict, idx: int) -> str:
    title = paper_entry["title"]
    authors = paper_entry["authors"]
    paper_string = f"{idx}. [{title}](#link{idx})\n"
    paper_string += f'**Authors:** {", ".join(authors)}\n'
    return paper_string


def render_md_string(papers):
    output_string = (
        "# Personalized Daily Arxiv Papers "
        + datetime.today().strftime("%m/%d/%Y")
        + "\nTotal relevant papers: "
        + str(len(papers))
        + "\n\n"
        + "Table of contents with paper titles:\n\n"
    )
    title_strings = [
        render_title_and_author(paper, i)
        for i, paper in enumerate(papers)
    ]
    output_string = output_string + "\n".join(title_strings) + "\n---\n"
    # render each paper
    paper_strings = [
        render_paper(paper, i) for i, paper in enumerate(papers)
    ]
    # join all papers into one string
    output_string = output_string + "\n".join(paper_strings)

    return output_string

pdb = get_papers_db("c")

papers = list()
with tqdm(total=len(pdb)) as pbar:
    for _, item in pdb.items():
        papers.append(
            {
                "title": item["title"],
                "abstract": item["summary"],
                "authors": [p["name"] for p in item["authors"]],
                "arxiv_id": item["_id"]
            }
        )

        pbar.update(1)
        if pbar.n % 10 == 0:
            break

output = render_md_string(papers)




