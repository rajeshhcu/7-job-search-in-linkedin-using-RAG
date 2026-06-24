from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent

def main(question):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(embedding=embeddings, index_name="resume-index")
    retriever = vectorstore.as_retriever()
    response = retriever.invoke(question)

    context = "".join([doc.page_content for doc in response])

    llm = ChatOpenAI(model="gpt-4o-mini")
    skills_response = llm.invoke(f"""
            Based on the following resume content, extract the top 5 most marketable technical skills 
            and the candidate's years of experience. Return as a comma-separated list of skills only.
            Resume: {context}
        """)
    skills = skills_response.content
    print(f"Extracted Skills: {skills}")

    tools = [TavilySearch(
        max_results=5,
        include_domains=["linkedin.com"]  # <-- restrict to LinkedIn only
    )]

    system_prompt = f"""You are a job search assistant. 
             Search LinkedIn for relevant job postings based on the candidate's skills {skills}.
             Always return actual LinkedIn job URLs in your final answer.
             
             Your search must strictly follow the below all rules.
             - Must return jobs posted in the USA.
             - Job posted not more than a week.
             - H1b visa sponsorship required.
             - Must match the skills in {skills}
             - Should not be staffing company.
             
             Format each result as:
             - Job Title at Company
               URL: <linkedin job url>
               Why it matches: <brief reason based on skills>
            """

    # Step 5: Create agent and executor
    agent = create_agent(model="openai:gpt-4o-mini", tools=tools, system_prompt=system_prompt)

    # Step 6: Run agent with skills injected into query
    search_query = f"""
            Search LinkedIn for senior engineer job openings matching these skills: {skills}.
            Find at least 5 job postings with direct LinkedIn URLs that the candidate can apply to.
            The candidate has 15+ years of experience in Python, Angular, AWS, and cloud technologies.
        """

    result = agent.invoke({"input": search_query})
    print("\n=== Suggested LinkedIn Jobs ===")
    print(result['messages'][-1].content)

if __name__ == "__main__":
    main("Candidate wants to apply for jobs in LinkedIn. Based on his resume, suggest suitable jobs for him in linkedin")