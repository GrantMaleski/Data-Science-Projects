#!/usr/bin/env python
# coding: utf-8

# Before you turn this problem in, make sure everything runs as expected. First, **restart the kernel** (in the menubar, select Kernel$\rightarrow$Restart) and then **run all cells** (in the menubar, select Cell$\rightarrow$Run All).
# 
# Make sure you fill in any place that says `YOUR CODE HERE` or "YOUR ANSWER HERE", as well as your name and collaborators below:

# ---

# # Coursework Assignment (COMP 42315)
# 
# The assignment consists of 3 questions. You are required to implement the programming solution containing your code and written answers that explain the implementation and justify the design. For more details, refer to the assignment brief on Blackboard Ultra page 
# 
# https://blackboard.durham.ac.uk/ultra/courses/_54354_1/outline/file/_1723127_1

# # Instructions (Question 1 and 2)
# 
# For Questions 1 and 2, you are asked to perform the following tasks based on the following target website, which contains artificial content designed for this assignment: https://sitescrape.awh.durham.ac.uk/comp42315/

# # Question 1 (35 marks)
# 
# Please design and implement a solution to crawl the publication title, year and author list of every unique publication record on the target website. Then, please create and display a table that contains these unique records. The table should consist of five columns: the row number in the table, publication title, year, author list, and the number of authors (hint: you will need to develop an algorithm to work this out). The records should be sorted first according to descending year values, then by descending number of author values, and finally by the titles from A to Z. Include the full final result in your Jupyter Notebook. 
# 
# [Explain your design and highlight any features in this question’s report part of your Jupyter Notebook in no more than 300 words. (35%)]

# ### Write your code in the following space.

# In[1]:


#Question 1

#Import Libraries
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

#Fetch page and save the html in soup
page = requests.get("https://sitescrape.awh.durham.ac.uk/comp42315/publicationfull_type_animationandgraphics.htm")
soup = BeautifulSoup(page.content,'html.parser')

#target the first p tag of the page, and then target the following one which holds the topics
p_tag = soup.find('p', class_='TextOption')

# target next ptag
next_p_tag = p_tag.find_next_sibling('p')

#target the links of each topic, so we can store them in a list
hrefs = [a['href'] for a in next_p_tag.find_all('a')]

topic_urls = ['https://sitescrape.awh.durham.ac.uk/comp42315/' + i for i in hrefs]

# The code above takes all of the urls except the first one, since it isnt hyperlinked...so we have to go to a different page and get the first url
second_url = topic_urls[0]

page = requests.get(second_url)
second = BeautifulSoup(page.content, 'html.parser')
    
twopage_p_tag = second.find('p', class_='TextOption')

twopage_next_p_tag = twopage_p_tag.find_next_sibling('p')

first_href = [a['href'] for a in twopage_next_p_tag.find_all('a')][0]

topic_urls.insert(0, 'https://sitescrape.awh.durham.ac.uk/comp42315/' + first_href)

#now we have all of our topic urls, we can begin going through each topic url and scraping all publications on each page
#before doing so, we can check to see if they have been added to unique_titles

records = []
unique_titles = []

for url in topic_urls:
    page = requests.get(url)
    each_page = BeautifulSoup(page.content,'html.parser')

    #find all publications on the site
    publications = (each_page.find_all("div", class_= "w3-cell-row"))


    #go through each publication and extract title, year, and authors
    for publication in publications:
        
        #check to see if it has been added before
        title = (publication.find("span", class_= "PublicationTitle").contents[0])
        if title not in unique_titles:
            
            #if it is uqique, append to both arrays
            each_publication = []
            each_publication.append(title)
            unique_titles.append(title)

            #use regular expression to target the year 
            year = (publication.find_all("span", class_= "TextSmall")[0].text)
            isolated_year = re.findall(r'\d+', year)[-1]
            clean_year = int(isolated_year)
            each_publication.append(clean_year)
            
            #split the authors up by the comma seperating them, remove the and at the end, and append to each_publication
            authors = (publication.find_all("span", class_= "TextSmall")[1].text)
            clean_authors = authors.replace(' and ', ', ')
            list_authors = clean_authors.split(",")
            list_authors = sorted(list_authors)
            each_publication.append(list_authors)
            
            #count authors in the list
            authors_count = len(list_authors)
            each_publication.append(authors_count)

            #append all the publications into the main records list
            records.append(each_publication)

#add the columns for the data and convert it to a df
df_1 = pd.DataFrame(records, columns =['Publication Title', 'Year', 'Author List', 'Number of Authors'])


# clean up the structure of the authors list and print it as a string instead of a list, so it looks better in the df
df_1['Author List'] = df_1['Author List'].str.join(',')

#sort the DataFrame accordingly
df_1 = df_1.sort_values(['Year', 'Number of Authors', 'Publication Title'], ascending=[False, False, True])

#insert the index column to show which row is being displayed
df_1.insert(0, 'Row Number', range(1, len(df_1) + 1))

#the indexes from the original data set were showing up so this helps replace those scattered numbers with the row number
df_1.set_index('Row Number', inplace=True)



(df_1)


# ### Write your description in the following space.

# Desing Outline: For Question 1, I designed the script to first fetch the target page which holds all of the publications and scrape the topic urls that hold all of the different publications. After I got the urls, the program itterates over them and scrapes the information for each publication on the page, such as the title, year of publication, authors. Once I have the author list, which I had to clean up and split by the comma, I can get the lenghth of that list giving me the author count. For some of these pieces of information, I had to utilitze regular expression which will help make this program. For example, the Year of the publication was often mixed in with the title. Regular expression allows me to target the integer(year) at the end of this string and save it as clean_year. All of this information goes into the records list, but before going in, it is checked to ensure that the title of the publication does not already exist in the unique_titles list. Once it is all in the rcords list, I convert it to a dataframe and add the columns for each value. The authors was initially presented as a nested list within the data frame so for presentation purposes, I converted it to a string. After this, I sorted the datafram accordlingly by 'Year', 'Number of Authors', 'Publication Title'. Once all this had, I inserted the row number, got rid of the old data frame indexes, and then printed the data frame. 

# # Question 2 (30 marks)
# 
# For this question, you should record the year, number of citations, topic, and number of Links, Downloads, and Online Resources (LDOR) for each journal paper listed on the scraping website and store these in a dataframe. Take care to filter the elements for uniqueness before producing the analysis. Produce a table showing the mean and variance of citations per journal publication in each topic and print it legibly in your submission. In Figure 1, you should show the mean and variance of citations per year across all journal publications for each topic. In Figure 2, you should show the number of LDOR against the number of citations for each journal publication. Each figure should be legible and have appropriate labels. 
# 
# [Explain your design and highlight any features in this question’s report part of your Jupyter Notebook in no more than 300 words. (30%)]

# ### Write your code in the following space.

# In[2]:


import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import matplotlib.pyplot as plt

url = "https://sitescrape.awh.durham.ac.uk/comp42315/publicationfull_type_animationandgraphics.htm"

page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')


p_tag = soup.find('p', class_='TextOption')

# Find the next p tag after the first one, since this is where all the topic urls are stored
next_p_tag = p_tag.find_next_sibling('p')


hrefs = [a['href'] for a in next_p_tag.find_all('a')]

topic_urls = ['https://sitescrape.awh.durham.ac.uk/comp42315/' + i for i in hrefs]

# The code above takes all of the urls except the first one, since it isnt hyperlinked...so we have to go to a different page and get the first url
second_url = topic_urls[0]

page = requests.get(second_url)
second = BeautifulSoup(page.content, 'html.parser')
    
twopage_p_tag = second.find('p', class_='TextOption')

twopage_next_p_tag = twopage_p_tag.find_next_sibling('p')

first_href = [a['href'] for a in twopage_next_p_tag.find_all('a')][0]

topic_urls.insert(0, 'https://sitescrape.awh.durham.ac.uk/comp42315/' + first_href)

#Now that we have all of our topic urls, Im going to document all of the topics that publications can fall under

#split the text using the '/' separator and remove leading/trailing whitespaces, as well as the unclean data in front of it all
topics = [topic.strip('Topic:\xa0\xa0\xa0') for topic in next_p_tag.text.split('/')]

#Okay now we have our urls and topics..let's go through all the topic urls to find the urls of each publication within them

all_topic_urls = []

# Iterate through topic URLs and topic list at the same time. This helps us keep track of which topic we are on for each url
for topic, topic_url in zip(topics, topic_urls):
    page = requests.get(topic_url)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    #we only want the lisitngs under journal papers, so we have to target the div under this which contains them
    
    publication_h2 = soup.find('h2', id='Journal Papers')

    if publication_h2:
        publication_journals = publication_h2.find_next('div')

        #extract URLs directly using list comprehension
        urls = [
            'https://sitescrape.awh.durham.ac.uk/comp42315/' + pub_div.find('a', class_='LinkButton')['href']
            for pub_div in publication_journals.find_all("div", class_="w3-cell-row")
        ]
    else:
        urls = []  # Empty list for topics with no journal paper links
    
    #store each url in a dictonary with the corresponding topic to know which topic it falls under
    all_topic_urls.append({
        'topic': topic,
        'topic_urls': urls
    })

#we will store all journal info in all_pubs, but will check if it exists already in unique_topic_url before adding it   
all_pubs = []
unique_topic_url = []

for topic_dict in all_topic_urls:
    topic = topic_dict['topic']
    topic_urls = topic_dict['topic_urls']

    # Iterate over each topic URL and keep track of the current index we are on
    for idx, topic_url in enumerate(topic_urls):
        
        #check to see if it exists already before iterating over and adding
        if topic_url not in unique_topic_url:
            unique_topic_url.append(topic_url)
            
            #create a dictionary that all the information can be store in
            each_pub = {"year": None, "citations": None, "topic": None, "LDOR Count": None}
            each_pub["topic"] = topic

            page = requests.get(topic_url)
            content = BeautifulSoup(page.content, 'html.parser')

            # focus in on the table elemnt of each page where a lot of the information is stored
            h2_element = content.find('h2', text='Links, Downloads and Online Resources')

            # Find the next div element after the h2
            targeted_div = h2_element.find_next('div')
            
            #set up a place for the ldor count to be stored, as well as a place to check if they have been added before
            hrefs = []
            unique_hrefs = []


            #count the number of links/ldor for each publication
            for a_tag in targeted_div.find_all('a', href=True):
                a_tags = a_tag['href']
                hrefs.append(a_tags)


                for i in hrefs:
                    if i not in unique_hrefs:
                        unique_hrefs.append(i) 


            if targeted_div.find_all('iframe'):
                # Check if the list is not empty
                iframe_src = targeted_div.find_all('iframe')[0]['src']
                unique_hrefs.append(iframe_src)
            else:
                pass

            each_pub["LDOR Count"] = len(unique_hrefs)

            # Extract Citation using regular expression
            citation_tag = content.find('h1').find('span').text

            citation_number = re.search(r'Citation:\s*(\d+)', citation_tag)

            if citation_number:
                number = citation_number.group(1)
                number = int(number)
                each_pub["citations"] = int(number)
            else:
                each_pub["citations"] = 0

            tables = content.find_all('table')

            for table in tables:
                table_text = table.get_text()

                # Extract year
                year_match = re.search(r'year={(\d+)}', table_text)
                if year_match:
                    year = year_match.group(1)
                    each_pub["year"] = int(year)
                else:
                    print("Year not found in this table.")

            all_pubs.append(each_pub)
            
        
        else:
            pass

df = pd.DataFrame(all_pubs)


print(df)


#----Figure 1 code - Mean Citations. There is so much data i'm going to split it the topics into two groups to help with visualization

#select 8 topics for each graph
topics_list = df['topic'].unique()
topics_per_graph = 8

#loop through topics and create separate graphs..I want to go from 0 to how many topics I have, with an increment of topics_per_graphh
for i in range(0, len(topics_list), topics_per_graph):
    #grab the first group from i to topics per group(which will be a step size of 8)
    selected_topics = topics_list[i:i+topics_per_graph]
    
    #filter data for the selected topics..we also have to have it in a list to then group by in the next step
    filtered_data = df[df['topic'].isin(selected_topics)]
    
    #group by year and topic and also calculate the mean
    grouped_data = filtered_data.groupby(['year', 'topic'])['citations'].mean().reset_index()

    #set up a pivot the table for plotting
    pivot_table = pd.pivot_table(grouped_data, values='citations', index='year', columns='topic')

    # plot it 
    pivot_table.plot(kind='bar', stacked=True)
    plt.title(f' Figure 1: Mean Citations per Year - Topics {i+1} to {i+topics_per_graph}')
    plt.xlabel('Year')
    plt.ylabel('Mean Citations')
    plt.legend(title='Topic', bbox_to_anchor=(1, 1), loc='upper left')

    plt.show()

#----Figure 1b Code

#This is really just the same as Figure 1 code, but I foucs on the variance instead of mean citations

#select 8 topics for each graph
topics_list = df['topic'].unique()
topics_per_graph = 8

#loop through topics and create separate graphs..I want to go from 0 to how many topics I have, with an increment of topics_per_graphh
for i in range(0, len(topics_list), topics_per_graph):
    #grab the first group from i to topics per group(which will be a step size of 8)
    selected_topics = topics_list[i:i+topics_per_graph]
    
    #filter data for the selected topics..we also have to have it in a list to then group by in the next step
    filtered_data = df[df['topic'].isin(selected_topics)]
    
    #group by year and topic and also calculate the mean
    grouped_data = filtered_data.groupby(['year', 'topic'])['citations'].var().reset_index()

    #set up a pivot the table for plotting
    pivot_table = pd.pivot_table(grouped_data, values='citations', index='year', columns='topic')

    # plot it 
    pivot_table.plot(kind='bar', stacked=True)
    plt.title(f' Figure 1b: Mean Citations per Year - Topics {i+1} to {i+topics_per_graph}')
    plt.xlabel('Year')
    plt.ylabel('Mean Citations')
    plt.legend(title='Topic', bbox_to_anchor=(1, 1), loc='upper left')

    plt.show()
    
    

#----Figure 2 Code - create a scatterdiagram for LDOR against citations


scatter = df[['LDOR Count', 'citations']].dropna()

#create the scatter plot

plt.scatter(scatter['LDOR Count'], scatter['citations'])

plt.title('Figure 2: Number of LDOR vs. Number of Citations for Each Journal Publication')
plt.xlabel('Number of LDOR')
plt.ylabel('Number of Citations')

plt.show()


# ### Write your description in the following space.

# The first part of the code follows the logic from question 1, meaning we first want to input the publications page and scrape for the topic urls and all topics that listings can fall under. Once we have gotten this information, I designed my code to only target journal publications by targeting the div under the Journals h2 tag. Once I isolated just the journal publications, I iterated over them to to target the url associated with them. I then saved the url with a corresponding topic, so that when I go into each url with BeautifulSoup, I know which topic it’s correlated with as the topic isn’t inside each journal page. Once I have all the journal urls, I can iterate over them using enumerate to help me keep track of the index I’m on so when I need to store the topic value, I can see what the topic value is for the current indexed url. The information from each publication page will be stored in a dictionary, but before initating this, I make sure that the topic_url we are on hasn’t already been scraped to help with uniqueness. I then go into each page using BeautifulSoup, and scrape the citation number, year of publication, and amount of links.  Once I have each page’s information in a single dictionary, I append it to my list all_pubs which will be then be converted to a data frame. For Figure 1, I split it up into subgraphs since there were so many categories. This helps reduce the clutter and helps visualize the mean of each topic for each year. The first subgraph shows the median and the second shows the variance. For Figure 2, I thought a scatter plot would best show the distribution of LDOR against citation.
# 

# # Instructions (Question 3)
# 
# For Question 3, you are asked to perform the task based on the target dataset (finance_dataset.csv), which you can download separately on Blackboard Ultra (refer link below). The file contains artificial content designed for this assignment.
# 
# https://blackboard.durham.ac.uk/ultra/courses/_54354_1/outline/file/_1723128_1
# 
# DO NOT CHANGE THE FILE NAME FOR THE CONSIDERED DATASET

# # Question 3 (35 marks)
# 
# The Cross-Sectional Asset Pricing dataset includes 210 features. The target variable is 'excessret', a firm's excess return between the current and the previous quarter. You are required to work on a subset that includes the 'defined features' and the 'target variable' by performing preprocessing (in the subset, there will be 11 features in total including target variable). You are required to extract the 'defined features' that are as indicated below: 
# 
# defined_columns = ['high52', 'mom12m', 'mom6m', 'maxret', 'mom12moffseason', 'realizedvol', 'idiovolaht', 'zerotrade', 'indretbig', 'returnskew']
# 
# Design and implement the solution to analyze the complex relationship between defined features and the firm's excess return between the current and the previous quarter. Highlight and visualise the attributes with the highest probabilistic relationship with the target variable. Justify the design choice and showcase the findings using an appropriate visualisation tool. 
# 
# [Explain your design and highlight any features in this question's report part of your Jupyter Notebook in no more than 400 words. (35%)]
# 

# ### Write your code in the following space.

# In[3]:


import csv
import pandas as pd
import scipy.stats
import numpy as np
import sklearn
import pgmpy as pg
import tabulate as tb
import networkx as nx
from sklearn.model_selection import train_test_split
from pgmpy.models import BayesianNetwork, BayesianModel
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import HillClimbSearch, BayesianEstimator
from pgmpy.models import BayesianModel
import networkx as nx
import matplotlib.pyplot as plt

finance = "finance_dataset.csv"

finance_df = pd.read_csv(finance)

finance_subset = finance_df[['excessret','high52', 'mom12m', 'mom6m', 'maxret', 'mom12moffseason', 'realizedvol', 'idiovolaht', 'zerotrade', 'indretbig', 'returnskew']]
finance_target = finance_df['excessret']

# See how many empty values there are
null_values = finance_subset.isnull().sum()


finance_subset.describe().transpose()


# I knew one of the biggest time consumping parts of this project was rebinning the data and their labels, 
# So i created a label generator to automatically generate the labels based on a desired number of bins
def label_generator(column_name, num_intervals):
    min_value = min(column_name)
    max_value = max(column_name)
    interval_size = (max_value - min_value) / num_intervals

    labels = []
    for i in range(num_intervals):
        lower_bound = min_value + i * interval_size
        upper_bound = lower_bound + interval_size

        label = f"{i + 1}: Between {lower_bound} and {upper_bound}"
        labels.append(label)

    return labels

#create a labels dictionary
labels = {}  

#for each column title, create a label in the dictionary using the label_generator
#I didnt use a for loop since each data set will need custom values to achieve the highest score
labels[finance_subset.columns[0]] = (label_generator(finance_subset[finance_subset.columns[0]], 2))
labels[finance_subset.columns[1]] = (label_generator(finance_subset[finance_subset.columns[1]], 2))
labels[finance_subset.columns[2]] = (label_generator(finance_subset[finance_subset.columns[2]], 2))
labels[finance_subset.columns[3]] = (label_generator(finance_subset[finance_subset.columns[3]], 2))
labels[finance_subset.columns[4]] = (label_generator(finance_subset[finance_subset.columns[4]], 4))
labels[finance_subset.columns[5]] = (label_generator(finance_subset[finance_subset.columns[5]], 2))
labels[finance_subset.columns[6]] = (label_generator(finance_subset[finance_subset.columns[6]], 3))
labels[finance_subset.columns[7]] = (label_generator(finance_subset[finance_subset.columns[7]], 5))
labels[finance_subset.columns[8]] = (label_generator(finance_subset[finance_subset.columns[8]], 1))
labels[finance_subset.columns[9]] = (label_generator(finance_subset[finance_subset.columns[9]], 3))
labels[finance_subset.columns[10]] = (label_generator(finance_subset[finance_subset.columns[10]], 3))



#discretize the data
def make_discrete(finance_subset, labels):
    discrete_df = pd.DataFrame()

    for col in finance_subset.columns:
        discrete_df[col] = pd.cut(finance_subset[col], bins=len(labels[col]), labels=labels[col], precision=2)

    discrete_df = discrete_df.astype('object')
    
    return discrete_df

cat_df = make_discrete(finance_subset, labels)




    
training_data, testing_data = train_test_split(cat_df, test_size = .2, random_state = 20)



hc = HillClimbSearch(data=training_data)
estimate = hc.estimate(scoring_method='k2score', max_iter=11.0, show_progress=True)
model = BayesianModel(estimate)

model.cpds = []
model.fit(data=training_data, estimator=BayesianEstimator, prior_type='BDeu', complete_samples_only=True)

model_edges = model.edges()

plt.figure(figsize=(14, 14))
G = nx.DiGraph()
G.add_edges_from(model_edges)


G.add_nodes_from(model.nodes)

#create subgraph for the node 'recourceState'
tt_g = G.subgraph(nodes=['recourceState'])

pos = nx.circular_layout(G)
DAG = G.to_directed()
nx.topological_sort(DAG)

nx.draw_networkx(G,
                pos=pos,
                with_labels=True,
                node_size=2000,  
                alpha=0.7,
                font_weight='bold',
                width=2.0
               )

# Visualize the subgraph
nx.draw_networkx(tt_g, pos=pos, with_labels=True, node_size=2000, alpha=0.7, font_weight='bold', width=2.0, node_color='r')

plt.show()

sub_g = G.subgraph(nodes = finance_subset.columns)

from pgmpy.metrics.metrics import correlation_score, log_likelihood_score, structure_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

accuracy_dict = {}

for column in testing_data:
    predict_data = testing_data.copy()
    predict_data.drop(column, axis = 1, inplace = True)
    y_pred= model.predict(predict_data)
    
    accuracy = accuracy_score(testing_data[column], y_pred)
    print(f'{column} Accuracy score: {accuracy}')
    accuracy_dict[column] = accuracy 
    
sum = 0
for v in accuracy_dict.values():
    sum += v
    
accuracy_dict['Average'] = sum/len(accuracy_dict.keys())

print('')
print(accuracy_dict)

f1 = correlation_score(model=model, data = testing_data, test = 'chi_square',significance_level = .05,score = f1_score, return_summary = False)
acc = correlation_score(model=model, data = testing_data, test = 'chi_square',significance_level = .05,score = accuracy_score, return_summary = False)
pr = correlation_score(model=model, data = testing_data, test = 'chi_square',significance_level = .05,score = precision_score, return_summary = False)
recall = correlation_score(model=model, data = testing_data, test = 'chi_square',significance_level = .05,score = recall_score, return_summary = False)
ls = log_likelihood_score(model= model, data = testing_data)
ss= structure_score(model=model, data = testing_data, scoring_method = 'bdeu')
print('')
print('f1 Score: '+ str(f1))
print('Accuracy Score: '+str(acc))
print('Precision Score: '+str(pr))
print('Recall Score: '+str(recall))
print('Log Likelihood Score: '+str(ls))
print('Structure Score: '+str(ss))
print(f'Check Model: {model.check_model()}\n')
for cpd in model.get_cpds():
    print(f'CPT of {cpd.variable}')
    print(cpd*100, '\n')


    
model.fit(data=training_data, estimator = BayesianEstimator)
bayes_model = BayesianNetwork(model)
bayes_model.fit(data = training_data,
               estimator = BayesianEstimator,
               prior_type = 'BDeu',
               complete_samples_only = True)


# In[4]:


for column in testing_data:
    predict_data = testing_data.copy()
    predict_data.drop(column, axis = 1, inplace = True)

testing_data.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Write your description in the following space.

# For the design of question 3, it mostly followed the logic of building a Bayesian Model outlined from lecture 4b and workshop 4. I chose a Bayesian Model because it's ability to model the complex relationships between variables, just as the questions asks,  with little data and it's ability to quantify our uncertainess on how these variables impact eachother. To conduct this model, I first subusetted the data to include the focused columns I want to work with. Then after transposing the columsn, I moved on to the main design improvement from the workshop: my label_generator function. I knew when building this model and trying to improve it, that changing the bin sizes would be a very time consuming aspect and I wanted to be able to create bin sizes and labels dynamically. So I built this function to help me later on when I'm trying to improve the model through small code changes to the bin sizes instead of rewritting lables and bin size numbers every time. After tinkering with the bin sizes a lot, I found that this combination of bin sizes in my code gets the highest precision and accuracy score while also revealing at least one connection to our target variable through my sub_g plot, showing maxret's causality on excessret. If I drop excessret's bin number to 1, both precision and accuracy go up about 10%, but our target variable is left with no meaningful insights, so I kept it this way. So, drawing conclusions from this, we can see that maxret has a notable impact on our target variables performance, along with an impact on realizedvol and returnskew. Furthermore, Idiovolaht is another variable carrying a large influence, showing probabalistic causality to three other variables, just as maxret is. 
