# pip install openai dotenv requests --proxy="http://lon3.sme.zscaler.net:443"

import random
import logging
from database_handler import DatabaseHandler
from chatgpt_handler import ChatGPTAnalyser
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if __name__ == "__main__":
    # Configuration
    DATABASE = "RothkoNLP"
    TEST_PATH = "R:\\Projects\\Rothko Business Objects\\Testing\\"

    # Create database handler instance
    db_handler = DatabaseHandler(DATABASE)
    
    # Create ChatGPT connection
    analyser = ChatGPTAnalyser()

    # Gather entities
    companies = db_handler.get_entity()

    if companies is None or companies.empty:
        logging.error("No companies found in the database.")

    # Randomly select 1000 companies (or fewer if the dataset is smaller)
    sample_size = min(5000, len(companies))
    selected_companies = companies.sample(n=sample_size) #, random_state=42)

    # DataFrame to store results
    article_data = []
    i = 0
    # For each company, gather searches and select one article to analyse
    for _, company in selected_companies.iterrows():
        try:
            i += 1
            logging.info(f"Company {i}/{sample_size}")
            
            company_sedol = company["EntityID"]

            # Gather searches
            search_results = db_handler.get_serpapi_output(id=0, entity_id=company_sedol, source='')
            # search_results = search_results[search_results["Article_Scrape_Fail"]==0] # successfully scraped articles

            if search_results is None or search_results.empty:
                logging.warning(f"No search results found for EntityID: {company_sedol}")
                continue  # Skip to the next company

            # Randomly select one article
            search_row = search_results.sample(n=1) #, random_state=42)
            article_id = search_row["Article_ID"].values[0]

            # Gather article
            # article = db_handler.get_article(article_id = int(article_id))

            # Store info
            company_name = company["EntityName"]
            article_title = search_row["Title"].values[0]
            article_snippet = search_row["Snippet"].values[0]
            # article_body = article["Body"].values[0]
            article_source = search_row["Source"].values[0]

            # ChatGPT article analysis
            result = analyser.analyse_article_relevance(company_name, article_title, article_snippet)

            if result == None: 
                logging.error(f"Failed to analyse article: {article_id}")
            
            # Store in list
            article_data.append({"Sedol": company_sedol, 
                                "Company Name": company_name,
                                "Article_ID": article_id, 
                                "Article_Source": article_source,
                                "Article_Title": article_title, 
                                "Article_Snippet": article_snippet, 
                                "Response_Subject": result['subject'], 
                                "Response_Mentioned": result['mentioned'],
                                "Response_Relevancy": result['relevance'], 
                                "Response_Usefulness": result['usefulness']})
            
        except Exception as e:
            logging.error(f"Unexpected error processing company {i+1}: {str(e)}")
            continue
            

    # Convert list to DataFrame
    articles_df = pd.DataFrame(article_data)

    articles_df.to_csv("articles_output_snippet.csv", index=False)
        