import urllib.parse
import urllib.request
import feedparser
from datetime import datetime, timedelta
import time


def search_arxiv(query, max_results=10, sort_by="submittedDate", sort_order="descending"):
    """
    Search ArXiv for papers matching the query.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
        sort_by: Sort criterion ('relevance', 'lastUpdatedDate', 'submittedDate')
        sort_order: Sort order ('ascending' or 'descending')

    Returns:
        List of paper dictionaries
    """
    base_url = "http://export.arxiv.org/api/query?"

    # Construct the query URL
    query_params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }

    url = base_url + urllib.parse.urlencode(query_params)

    # Fetch and parse the results
    response = urllib.request.urlopen(url)
    feed = feedparser.parse(response)

    papers = []
    for entry in feed.entries:
        paper = {
            "title": entry.title,
            "authors": [author.name for author in entry.authors],
            "published": entry.published,
            "updated": entry.updated,
            "summary": entry.summary,
            "link": entry.link,
            "pdf_link": next((link.href for link in entry.links if link.type == "application/pdf"), None),
            "categories": [tag.term for tag in entry.tags],
        }
        papers.append(paper)

    return papers


def format_paper(paper, index):
    """Format a paper for display."""
    print(f"\n{'=' * 80}")
    print(f"Paper {index}: {paper['title']}")
    print(f"Authors: {', '.join(paper['authors'][:3])}{' et al.' if len(paper['authors']) > 3 else ''}")
    print(f"Published: {paper['published'][:10]}")
    print(f"Categories: {', '.join(paper['categories'][:3])}")
    print(f"Link: {paper['link']}")
    if paper["pdf_link"]:
        print(f"PDF: {paper['pdf_link']}")
    print(f"\nAbstract: {paper['summary'][:500]}...")


def search_multiple_topics():
    """Search for papers on multiple topics related to generation and diffusion."""

    # Define search queries for different topics
    searches = [
        {
            "name": "Image Generation & Diffusion",
            "query": '(ti:diffusion OR abs:diffusion) AND (ti:"image generation" OR abs:"image generation" OR ti:"text-to-image" OR abs:"text-to-image")',
            "max_results": 5,
        },
        {
            "name": "Video Generation",
            "query": '(ti:"video generation" OR abs:"video generation" OR ti:"video diffusion" OR abs:"video diffusion" OR ti:"text-to-video" OR abs:"text-to-video")',
            "max_results": 5,
        },
        {
            "name": "World Models",
            "query": '(ti:"world model" OR abs:"world model" OR ti:"world models" OR abs:"world models") AND (ti:generation OR abs:generation OR ti:learning OR abs:learning)',
            "max_results": 5,
        },
        {
            "name": "Latest Diffusion Techniques",
            "query": '(ti:diffusion OR abs:diffusion) AND (ti:transformer OR ti:flow OR ti:consistency OR ti:rectified OR ti:distillation OR abs:"score matching")',
            "max_results": 5,
        },
        {
            "name": "Diffusion Model Improvements",
            "query": '(ti:"diffusion model" OR abs:"diffusion model") AND (ti:efficient OR ti:fast OR ti:improved OR ti:novel OR abs:acceleration)',
            "max_results": 5,
        },
    ]

    all_results = {}

    for search in searches:
        print(f"\n{'#' * 80}")
        print(f"Searching for: {search['name']}")
        print(f"Query: {search['query']}")
        print(f"{'#' * 80}")

        try:
            papers = search_arxiv(
                search["query"], max_results=search["max_results"], sort_by="submittedDate", sort_order="descending"
            )

            all_results[search["name"]] = papers

            if papers:
                print(f"\nFound {len(papers)} papers:")
                for i, paper in enumerate(papers, 1):
                    format_paper(paper, i)
            else:
                print("No papers found for this query.")

        except Exception as e:
            print(f"Error searching for {search['name']}: {e}")

        # Be respectful of the API rate limits
        time.sleep(3)

    return all_results


def search_custom_query(query, max_results=10):
    """
    Search with a custom query string.

    Example queries:
    - 'ti:"stable diffusion" AND cat:cs.CV' - Papers with "stable diffusion" in title in Computer Vision
    - 'au:Rombach' - Papers by author Rombach
    - 'all:transformer AND all:diffusion' - Papers mentioning both transformer and diffusion
    """
    print(f"\nSearching for: {query}")
    papers = search_arxiv(query, max_results=max_results)

    if papers:
        print(f"\nFound {len(papers)} papers:")
        for i, paper in enumerate(papers, 1):
            format_paper(paper, i)
    else:
        print("No papers found.")

    return papers


def get_recent_papers_by_date(days_back=7):
    """Get papers from the last N days on generation topics."""

    # Calculate the date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    # Format dates for ArXiv query (YYYYMMDD)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    # Construct date-based query
    base_query = f"submittedDate:[{start_str}0000 TO {end_str}2359]"

    # Add topic filters
    topic_query = '(all:diffusion OR all:"image generation" OR all:"video generation" OR all:"world model")'

    full_query = f"{base_query} AND {topic_query}"

    print(f"\nSearching for papers from the last {days_back} days")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    papers = search_arxiv(full_query, max_results=20)

    if papers:
        print(f"\nFound {len(papers)} recent papers:")
        for i, paper in enumerate(papers, 1):
            format_paper(paper, i)
    else:
        print("No recent papers found.")

    return papers


if __name__ == "__main__":
    print("ArXiv Paper Search for Image/Video Generation and Diffusion Models")
    print("=" * 80)

    # Run the multi-topic search
    print("\n1. SEARCHING MULTIPLE TOPICS")
    results = search_multiple_topics()

    # Example of custom query
    print("\n2. CUSTOM QUERY EXAMPLE")
    custom_papers = search_custom_query('ti:"latent diffusion" AND cat:cs.CV', max_results=3)

    # Get papers from the last week
    print("\n3. RECENT PAPERS (LAST 7 DAYS)")
    recent_papers = get_recent_papers_by_date(days_back=7)

    # Save results to file (optional)
    print("\n" + "=" * 80)
    save_option = input("\nWould you like to save the results to a file? (y/n): ")

    if save_option.lower() == "y":
        import json

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"arxiv_papers_{timestamp}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "search_date": datetime.now().isoformat(),
                    "results": results,
                    "custom_search": custom_papers,
                    "recent_papers": recent_papers,
                },
                f,
                indent=2,
                default=str,
            )

        print(f"Results saved to {filename}")
