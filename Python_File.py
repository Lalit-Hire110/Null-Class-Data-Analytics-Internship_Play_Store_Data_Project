#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import webbrowser
import os
import warnings
warnings.filterwarnings("ignore")


# In[2]:


nltk.download('vader_lexicon')


# In[3]:


apps_df=pd.read_csv('Play Store Data.csv')
reviews_df=pd.read_csv('User Reviews.csv')


# In[4]:


apps_df.head()


# In[5]:


reviews_df.head()


# In[6]:


#pd.read_csv() : csv files
#pd.read_excel() : excel files
#pd.read_sql() : SQL Databases
#pd.read_json() : JSON Files


# In[7]:


#df.isnull() : Missing values
#df.dropna() : Removes rows and columns that contain the missing values
#df.fillna() : Fills missing values


# In[8]:


#df.duplicated() : Identifies duplicates
#df.drop_duplicates() : Removes duplicate rows


# In[9]:


#Step 2 : Data Cleaning
apps_df = apps_df.dropna(subset=['Rating'])
for column in apps_df.columns :
    apps_df[column].fillna(apps_df[column].mode()[0],inplace=True)
apps_df.drop_duplicates(inplace=True)
apps_df=apps_df=apps_df[apps_df['Rating']<=5]
reviews_df.dropna(subset=['Translated_Review'],inplace=True)


# In[10]:


apps_df.dtypes


# In[11]:


#Convert the Installs columns to numeric by removing commas and +
apps_df['Installs']=apps_df['Installs'].str.replace(',','').str.replace('+','').astype(int)

#Convert Price column to numeric after removing $
apps_df['Price']=apps_df['Price'].str.replace('$','').astype(float)


# In[12]:


apps_df.dtypes


# In[13]:


merged_df=pd.merge(apps_df,reviews_df,on='App',how='inner')


# In[14]:


merged_df.head()


# In[15]:


def convert_size(size):
    if 'M' in size:
        return float(size.replace('M',''))
    elif 'k' in size:
        return float(size.replace('k',''))/1024
    else:
        return np.nan
apps_df['Size']=apps_df['Size'].apply(convert_size)


# In[16]:


apps_df


# In[17]:


#Lograrithmic
apps_df['Log_Installs']=np.log(apps_df['Installs'])


# In[18]:


apps_df['Reviews']=apps_df['Reviews'].astype(int)


# In[19]:


apps_df['Log_Reviews']=np.log(apps_df['Reviews'])


# In[20]:


apps_df.dtypes


# In[21]:


def rating_group(rating):
    if rating >= 4:
        return 'Top rated app'
    elif rating >=3:
        return 'Above average'
    elif rating >=2:
        return 'Average'
    else:
        return 'Below Average'
apps_df['Rating_Group']=apps_df['Rating'].apply(rating_group)


# In[22]:


#Revenue column
apps_df['Revenue']=apps_df['Price']*apps_df['Installs']


# In[23]:


sia = SentimentIntensityAnalyzer()


# In[24]:


#Polarity Scores in SIA
#Positive, Negative, Neutral and Compound: -1 - Very negative ; +1 - Very positive


# In[25]:


review = "This app is amazing! I love the new features."
sentiment_score= sia.polarity_scores(review)
print(sentiment_score)


# In[26]:


review = "This app is very bad! I hate the new features."
sentiment_score= sia.polarity_scores(review)
print(sentiment_score)


# In[27]:


review = "This app is okay."
sentiment_score= sia.polarity_scores(review)
print(sentiment_score)


# In[28]:


reviews_df['Sentiment_Score']=reviews_df['Translated_Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])


# In[29]:


reviews_df.head()


# In[30]:


apps_df['Last Updated']=pd.to_datetime(apps_df['Last Updated'],errors='coerce')


# In[31]:


apps_df['Year']=apps_df['Last Updated'].dt.year


# In[32]:


import random

# Listing countries so wwe can use them to plot chorograph with more visuals. due to we don't have any coloumn which doesn't provides any location info contries have been taken randomly. 
countries = [
    "United States", "India", "United Kingdom", "Germany", "Canada",
    "France", "Australia", "China", "Brazil", "Japan", "Russia", "South Africa"
]

# Step 2: Add a 'Country' column to dataset.
apps_df["Country"] = [random.choice(countries) for _ in range(len(apps_df))]


# In[33]:


apps_df.head()


# In[34]:


html_files_path="./"
if not os.path.exists(html_files_path):
    os.makedirs(html_files_path)


# In[35]:


plot_containers=""


# In[36]:


# Save each Plotly figure to an HTML file
def save_plot_as_html(fig, filename, insight):
    global plot_containers
    filepath = os.path.join(html_files_path, filename)
    html_content = pio.to_html(fig, full_html=False, include_plotlyjs='inline')
    # Append the plot and its insight to plot_containers
    plot_containers += f"""
    <div class="plot-container" id="{filename}" onclick="openPlot('{filename}')">
        <div class="plot">{html_content}</div>
        <div class="insights">{insight}</div>
    </div>
    """
    fig.write_html(filepath, full_html=False, include_plotlyjs='inline')


# In[37]:


plot_width=400
plot_height=300
plot_bg_color='black'
text_color='white'
title_font={'size':16}
axis_font={'size':12}


# In[38]:


#Figure 1
category_counts=apps_df['Category'].value_counts().nlargest(10)
fig1=px.bar(
    x=category_counts.index,
    y=category_counts.values,
    labels={'x':'Category','y':'Count'},
    title='Top Categories on Play Store',
    color=category_counts.index,
    color_discrete_sequence=px.colors.sequential.Plasma,
    width=400,
    height=300
)
fig1.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig1.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig1,"Category Graph 1.html","The top categories on the Play Store are dominated by tools, entertainment, and productivity apps")
            


# In[39]:


#Figure 2
type_counts=apps_df['Type'].value_counts()
fig2=px.pie(
    values=type_counts.values,
    names=type_counts.index,
    title='App Type Distribution',
    color_discrete_sequence=px.colors.sequential.RdBu,
    width=400,
    height=300
)
fig2.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig1.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig2,"Type Graph 2.html","Most apps on the Playstore are free, indicating a strategy to attract users first and monetize through ads or in app purchases")


# In[40]:


#Figure 3
fig3=px.histogram(
    apps_df,
    x='Rating',
    nbins=20,
    title='Rating Distribution',
    color_discrete_sequence=['#636EFA'],
    width=400,
    height=300
)
fig3.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig1.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig3,"Rating Graph 3.html","Ratings are skewed towards higher values, suggesting that most apps are rated favorably by users")


# In[41]:


#Figure 4
sentiment_counts=reviews_df['Sentiment_Score'].value_counts()
fig4=px.bar(
    x=sentiment_counts.index,
    y=sentiment_counts.values,
    labels={'x':'Sentiment Score','y':'Count'},
    title='Sentiment Distribution',
    color=sentiment_counts.index,
    color_discrete_sequence=px.colors.sequential.RdPu,
    width=400,
    height=300
)
fig4.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig1.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig4,"Sentiment Graph 4.html","Sentiments in reviews show a mix of positive and negative feedback, with a slight lean towards positive sentiments")


# In[42]:


#Figure 5
installs_by_category=apps_df.groupby('Category')['Installs'].sum().nlargest(10)
fig5=px.bar(
    x=installs_by_category.index,
    y=installs_by_category.values,
    orientation='h',
    labels={'x':'Installs','y':'Category'},
    title='Installs by Category',
    color=installs_by_category.index,
    color_discrete_sequence=px.colors.sequential.Blues,
    width=400,
    height=300
)
fig5.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig1.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig5,"Installs Graph 5.html","The categories with the most installs are social and communication apps, reflecting their broad appeal and daily usage")


# In[43]:


# Updates Per Year Plot
updates_per_year = apps_df['Last Updated'].dt.year.value_counts().sort_index()
fig6 = px.line(
    x=updates_per_year.index,
    y=updates_per_year.values,
    labels={'x': 'Year', 'y': 'Number of Updates'},
    title='Number of Updates Over the Years',
    color_discrete_sequence=['#AB63FA'],
    width=plot_width,
    height=plot_height
)
fig6.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(fig6, "Updates Graph 6.html", "Updates have been increasing over the years, showing that developers are actively maintaining and improving their apps.")


# In[44]:


#Figure 7
revenue_by_category=apps_df.groupby('Category')['Revenue'].sum().nlargest(10)
fig7=px.bar(
    x=installs_by_category.index,
    y=installs_by_category.values,
    labels={'x':'Category','y':'Revenue'},
    title='Revenue by Category',
    color=installs_by_category.index,
    color_discrete_sequence=px.colors.sequential.Greens,
    width=400,
    height=300
)
fig7.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig1.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig7,"Revenue Graph 7.html","Categories such as Business and Productivity lead in revenue generation, indicating their monetization potential")


# In[45]:


#Figure 8
genre_counts=apps_df['Genres'].str.split(';',expand=True).stack().value_counts().nlargest(10)
fig8=px.bar(
    x=genre_counts.index,
    y=genre_counts.values,
    labels={'x':'Genre','y':'Count'},
    title='Top Genres',
    color=installs_by_category.index,
    color_discrete_sequence=px.colors.sequential.OrRd,
    width=400,
    height=300
)
fig8.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig1.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig8,"Genre Graph 8.html","Action and Casual genres are the most common, reflecting users' preference for engaging and easy-to-play games")


# In[46]:


#Figure 9
fig9=px.scatter(
    apps_df,
    x='Last Updated',
    y='Rating',
    color='Type',
    title='Impact of Last Update on Rating',
    color_discrete_sequence=px.colors.qualitative.Vivid,
    width=400,
    height=300
)
fig9.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig1.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig9,"Update Graph 9.html","The Scatter Plot shows a weak correlation between the last update and ratings, suggesting that more frequent updates dont always result in better ratings.")


# In[47]:


#Figure 10
fig10=px.box(
    apps_df,
    x='Type',
    y='Rating',
    color='Type',
    title='Rating for Paid vs Free Apps',
    color_discrete_sequence=px.colors.qualitative.Pastel,
    width=400,
    height=300
)
fig10.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={'size':12}),
    margin=dict(l=10,r=10,t=30,b=10)
)
#fig1.update_traces(marker=dict(pattern=dict(line=dict(color='white',width=1))))
save_plot_as_html(fig10,"Paid Free Graph 10.html","Paid apps generally have higher ratings compared to free apps, suggesting that users expect higher quality from apps they pay for")


# ## for the task given in the Internship
# ### I have completed all 3 given tasks, integreting them with the dashboard we have build while being on training.

# In[49]:


import pytz
from datetime import datetime

# Function to check if the current time falls within a specific range
def is_within_time_range(start_hour, end_hour, timezone="Asia/Kolkata"):
    tz = pytz.timezone(timezone)
    current_time = datetime.now(tz).time()
    return start_hour <= current_time.hour < end_hour


# #### Task 1:
# ##### Create a scatter plot to visualize the relationship between revenue and the number of installs for paid apps only. Add a trendline to show the correlation and color-code the points based on app categories.

# In[51]:


fig11 = px.scatter(
    apps_df,
    x="Installs",
    y="Revenue",
    color="Category",
    title="Revenue vs Installs for Paid Apps",
    labels={"Installs": "Number of Installs", "Revenue": "Revenue ($)"},
    trendline="ols",
    hover_data=["App"],
    width=plot_width,
    height=plot_height
)
fig11.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)
save_plot_as_html(
    fig11,
    "Scatter_Revenue_vs_Installs.html",
    "This plot shows the relationship between revenue and the number of installs for paid apps, categorized by app type."
)


# #### Task 2:
# ##### Create an interactive choropleth map using Plotly to visualize global installs by categories . Apply filters to show data for only the top 5 app categories and highlight categories where the number of installs exceeds 1 million and App category should not start with character “A” , “C” , “G” and “S” . this graph should work only between 6PM IST to 8 PM IST apart from that time we should not show this graph in dashboard itself.

# In[56]:


if is_within_time_range(18, 20):  # Only display between 6 PM IST and 8 PM IST
    # Filter the dataframe for relevant data
    choropleth_df = apps_df[
        (apps_df["Category"].str[0].isin(["A", "C", "G", "S"]) == False) &
        (apps_df["Installs"] > 1_000_000)
    ]

    # Identify the top 5 categories by total installs
    top_categories = (
        choropleth_df.groupby("Category")["Installs"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .index
    )
    # Filter dataframe for the top 5 categories
    choropleth_df = choropleth_df[choropleth_df["Category"].isin(top_categories)]

    # Create the choropleth map
    fig12 = px.choropleth(
        choropleth_df,
        locations="Country",  # Ensure this column contains valid country names
        locationmode="country names",  # Use country names for mapping
        color="Installs",
        hover_name="Category",
        title="Global Installs by Country",
        color_continuous_scale=px.colors.sequential.Turbo,  # Vibrant color scale
        width=plot_width,
        height=plot_height
    )

    # Update layout for better visualization
    fig12.update_layout(
        coloraxis_colorbar=dict(
            title="Installs",
            tickprefix="",
            ticks="outside"
        ),
        plot_bgcolor=plot_bg_color,
        paper_bgcolor=plot_bg_color,
        font_color=text_color,
        title_font=title_font,
        margin=dict(l=10, r=10, t=30, b=10)
    )

    # Save the plot as an HTML file
    save_plot_as_html(
        fig12,
        "Choropleth_Global_Installs_with_Countries.html",
        "This map highlights the global installs for top categories by country, with categories having over 1 million installs."
    )
else:
    print("Task 2: Choropleth map is not available at this time.")


# In[ ]:


# if is_within_time_range(18, 20):  # Only display between 6 PM IST and 8 PM IST
#     choropleth_df = apps_df[
#         (apps_df["Category"].str[0].isin(["A", "C", "G", "S"]) == False) &
#         (apps_df["Installs"] > 1_000_000)
#     ]

#     top_categories = (
#         choropleth_df.groupby("Category")["Installs"]
#         .sum()
#         .sort_values(ascending=False)
#         .head(5)
#         .index
#     )
#     choropleth_df = choropleth_df[choropleth_df["Category"].isin(top_categories)]

#     fig12 = px.choropleth(
#         choropleth_df,
#         locations="Country",  # Use the new Country column
#         locationmode="country names",  # Map by country names
#         color="Installs",
#         hover_name="Category",
#         title="Global Installs by Country",
#         color_continuous_scale=px.colors.sequential.Plasma,
#         width=plot_width,
#         height=plot_height
#     )
#     fig12.update_layout(
#         plot_bgcolor=plot_bg_color,
#         paper_bgcolor=plot_bg_color,
#         font_color=text_color,
#         title_font=title_font,
#         margin=dict(l=10, r=10, t=30, b=10)
#     )
#     save_plot_as_html(
#         fig12,
#         "Choropleth_Global_Installs_with_Countries.html",
#         "This map highlights the global installs for top categories by country, with categories having over 1 million installs."
#     )
# else:
#     print("Task 2: Choropleth map is not available at this time.")


# #### Task 3:
# ##### Plot a bubble chart to analyze the relationship between app size (in MB) and average rating, with the bubble size representing the number of installs. Include a filter to show only apps with a rating higher than 3.5 and that belong to the "Games" category and installs should be more than 50k as well as this graph should work only between 5 PM IST to 7 PM IST apart from that time we should not show this graph in dashboard itself.

# In[58]:


if is_within_time_range(17, 19):  # Only display between 5 PM IST and 7 PM IST
    bubble_df = apps_df[
        (apps_df["Category"] == "GAME") &
        (apps_df["Rating"] > 3.5) &
        (apps_df["Installs"] > 50_000)
    ]

    fig13 = px.scatter(
        bubble_df,
        x="Size",
        y="Rating",
        size="Installs",
        color="Genres",
        title="App Size vs Rating for Games",
        labels={"Size": "App Size (MB)", "Rating": "Average Rating"},
        hover_data=["App"],
        width=plot_width,
        height=plot_height
    )
    fig13.update_layout(
        plot_bgcolor=plot_bg_color,
        paper_bgcolor=plot_bg_color,
        font_color=text_color,
        title_font=title_font,
        xaxis=dict(title_font=axis_font),
        yaxis=dict(title_font=axis_font),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    save_plot_as_html(
        fig13,
        "Bubble_Size_vs_Rating.html",
        "This bubble chart visualizes the relationship between app size and rating for games with more than 50k installs and a rating above 3.5."
    )
else:
    print("Task 3: Bubble chart is not available at this time.")


# In[60]:


plot_containers_split=plot_containers.split('</div>')


# In[62]:


if len(plot_containers_split) > 1:
    final_plot=plot_containers_split[-2]+'</div>'
else:
    final_plot=plot_containers


# In[64]:


dashboard_html= """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name=viewport" content="width=device-width,initial-scale-1.0">
    <title> Google Play Store Review Analytics</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #333;
            color: #fff;
            margin: 0;
            padding: 0;
        }}
        .header {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background-color: #444
        }}
        .header img {{
            margin: 0 10px;
            height: 50px;
        }}
        .container {{
            display: flex;
            flex-wrap: wrap;
            justify_content: center;
            padding: 20px;
        }}
        .plot-container {{
            border: 2px solid #555
            margin: 10px;
            padding: 10px;
            width: {plot_width}px;
            height: {plot_height}px;
            overflow: hidden;
            position: relative;
            cursor: pointer;
        }}
        .insights {{
            display: none;
            position: absolute;
            right: 10px;
            top: 10px;
            background-color: rgba(0,0,0,0.7);
            padding: 5px;
            border-radius: 5px;
            color: #fff;
        }}
        .plot-container: hover .insights {{
            display: block;
        }}
        </style>
        <script>
            function openPlot(filename) {{
                window.open(filename, '_blank');
                }}
        </script>
    </head>
    <body>
        <div class= "header">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Logo_2013_Google.png/800px-Logo_2013_Google.png" alt="Google Logo">
            <h1>Google Play Store Reviews Analytics</h1>
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Google_Play_Store_badge_EN.svg/1024px-Google_Play_Store_badge_EN.svg.png" alt="Google Play Store Logo">
        </div>
        <div class="container">
            {plots}
        </div>
    </body>
    </html>
    """


# In[66]:


final_html=dashboard_html.format(plots=plot_containers,plot_width=plot_width,plot_height=plot_height)


# In[68]:


dashboard_path=os.path.join(html_files_path,"web page.html")


# In[70]:


with open(dashboard_path, "w", encoding="utf-8") as f:
    f.write(final_html)


# In[72]:


webbrowser.open('file://'+os.path.realpath(dashboard_path))

