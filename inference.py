from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field


class JobResult(BaseModel):
    job_title: str = Field(description="Job title at company")
    job_url: str = Field(description="Direct LinkedIn job URL")
    why_it_matches: str = Field(description="Specific skills from candidate profile that match")
    job_location: str = Field(description="City, State, USA")
    is_usa: bool = Field(description="True if job is located in the USA")
    posted_within_week: bool = Field(description="True if job was posted within the last 7 days on June 23, 2026")
    offers_h1b: bool = Field(description="True if job offers H1B visa sponsorship")


class Result(BaseModel):
    jobs: list[JobResult] = Field(description="List of validated job postings")


RULES = """
STRICT FILTERING RULES — discard any job that fails even one:
1. Location must be in the USA (remote-US counts, international does NOT).
2. Job must have been posted within the last 7 days.
3. Job must explicitly mention H1B visa sponsorship.
4. Job must still be accepting applications.
5. Job must match the candidate's extracted skills.
"""


def validate_jobs(result: Result) -> list[JobResult]:
    valid = []
    for job in result.jobs:
        if (
            job.is_usa
            and job.posted_within_week
            and job.offers_h1b
            and "linkedin.com" in job.job_url
        ):
            valid.append(job)
        else:
            print(f"[FILTERED OUT] {job.job_title} — failed rule validation")
    return valid


def main(question: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(embedding=embeddings, index_name="resume-index")
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])

    llm = ChatOpenAI(model="gpt-4o-mini")
    skills_response = llm.invoke(
        f"""Extract the top 5 most marketable technical skills from this resume.
        Return ONLY a comma-separated list, nothing else.
        Resume: {context}"""
    )
    skills = skills_response.content.strip()
    print(f"Extracted Skills: {skills}")

    tool = TavilySearchResults(
        max_results=5,
        search_depth="basic",
        include_answer=False,
        include_raw_content=False,
    )

    system_prompt = f"""You are a job search assistant. Search LinkedIn ONCE for senior engineer jobs.

Candidate skills: {skills}

{RULES}

IMPORTANT INSTRUCTIONS:
- Run ONE search only, then compile results immediately. Do NOT loop or retry.
- Use this exact search query:
  site:linkedin.com/jobs senior engineer ({skills}) H1B sponsorship USA posted this week
- Return exactly 5 jobs. If fewer than 5 pass the filters, return however many do.
- Set all boolean fields accurately for each job.
"""

    agent = create_react_agent(
        model="openai:gpt-4o-mini",
        tools=[tool],
        prompt=system_prompt,
        response_format=Result,
    )

    search_query = (
        f"Find senior engineer jobs on LinkedIn matching: {skills}. "
        f"Requirements: USA only, posted this week, H1B sponsorship. "
        f"Run ONE search and return 5 results immediately."
    )

    print("Searching... (this may take 15-30 seconds)")
    response = agent.invoke(
        {"messages": [("user", search_query)]},
        config={"recursion_limit": 5}
    )

    raw_result: Result = response["structured_response"]
    valid_jobs = validate_jobs(raw_result)

    print(f"\n=== LinkedIn Jobs ({len(valid_jobs)} passed all filters) ===")
    for job in valid_jobs:
        print("=" * 60)
        print(f"  Job Title    : {job.job_title}")
        print(f"  LinkedIn URL : {job.job_url}")
        print(f"  Location     : {job.job_location}")
        print(f"  Why it fits  : {job.why_it_matches}")
        print("=" * 60)

    if not valid_jobs:
        print("No jobs passed all filters. Try broadening the search.")


if __name__ == "__main__":
    main(
        "Candidate wants to apply for jobs on LinkedIn. "
        "Based on his resume, suggest suitable jobs."
    )