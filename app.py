import streamlit as st
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import keepa

st.set_page_config(
    page_title="Real-Time Data Science Dashboard",
    page_icon="✅",
    layout="wide",
)


accesskey = 'c22mt3o8utp5jj1rnlobbptmc0gdfvjt3n6dg3p2gd2pi34ajj539vkrhluinu8s' # enter real access key here
api = keepa.Keepa(accesskey)

st.write("get_data dan gelen veri:", st.session_state.result_data)


with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# with open("style.css") as f:
#         st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def product_time(min_list):
    """Convert raw keepa time to datetime!"""
    keepa_time = datetime.datetime(2011, 1 ,1)
    return [keepa_time + datetime.timedelta(minutes = i) for i in min_list]

def get_asin():
    return st.text_input('Enter some text')
# Using object notation


# fig = plt.figure(figsize=(12,7))
# plt.plot(x, y)

# # Türevin sıfır olduğu noktaları bulma
# dy = np.diff(y)
# tepe_noktalari = np.where(np.diff(np.sign(dy)) < 0)[0] + 1  # Tepe noktaları
# cukur_noktalari = np.where(np.diff(np.sign(dy)) > 0)[0] + 1  # Çukur noktaları

# # Tepe noktalarını grafiğe işaretleme
# plt.plot(x[tepe_noktalari], y[tepe_noktalari], 'ro', label='Tepe Noktaları')
# # Çukur noktalarını grafiğe işaretleme
# plt.plot(x[cukur_noktalari], y[cukur_noktalari], 'bo', label='Çukur Noktaları')

# plt.legend()
# plt.show()

# print(y[tepe_noktalari].mean(),
# y[cukur_noktalari].mean(),y[tepe_noktalari].argmax())

# print(tepe_noktalari)

# Using "with" notation
with st.sidebar:
    add_radio = st.radio(
        "B001LDKAX2",
        ("Standard (5-15 days)", "Express (2-5 days)")
    )
with st.sidebar:
    asin = get_asin()
    if st.button("Press me"):
            st.success("Analyzing")


#####  col1, col2 = st.columns([5,1])




st.write()
tab_x, tab_y, tab_z = st.tabs(["graphs"," analysis","images"])
asin = int(asin)

if asin:  #"B001LDKAX2"
    products = api.query(st.session_state.result_data.iloc[asin]["asin"],stats=90,offers=100) # returns list of product data
    
    keepa_time = datetime.datetime(2011, 1 ,1)
    bb_stats = []
    for t,j in st.session_state.result_data.iloc[asin]["stats_parsed"]["buyBoxStats"].items():
        if t == "ATVPDKIKX0DER":
            bb_stats.append(("amz", round(j["percentageWon"]),round((int(j["avgPrice"]) / 100), 2), j["avgNewOfferCount"], keepa_time + datetime.timedelta(minutes = j['lastSeen'])))
        elif j["isFBA"]:
            bb_stats.append(("fba", round(j["percentageWon"]),round((int(j["avgPrice"]) / 100), 2), j["avgNewOfferCount"],  keepa_time + datetime.timedelta(minutes = j['lastSeen'])))
        else:
            bb_stats.append(("fbm", round(j["percentageWon"]),round((int(j["avgPrice"]) / 100), 2), j["avgNewOfferCount"],  keepa_time + datetime.timedelta(minutes = j['lastSeen'])))

    bb_stats_df = pd.DataFrame(bb_stats,columns = ["sellerType",'percentageWon', 'avgPrice', 'avgNewOfferCount', 'lastSeen'])
    bb_stats_df = bb_stats_df.sort_values(["percentageWon"],ascending=False).reset_index(drop=True)

    
    imagesCSV = st.session_state.result_data.iloc[asin]['imagesCSV'].split(",")

    sales = pd.DataFrame()
    sales["datetime"]  = product_time(products[0]['csv'][3][::-1][1::2])
    sales["sales_rank"]  = products[0]['csv'][3][::-1][::2]
    sales["sales_rank"] = sales["sales_rank"].apply(lambda x: None if x == -1 else x)
    sales.set_index("datetime")

    last_month = datetime.datetime.now() - datetime.timedelta(days = 365)
    sales = sales[sales["datetime"].apply(lambda x: True if x >= last_month else False)]

    sales["date"] = sales.datetime.dt.date

    liste22 = []
    for i in sales.date:
        liste22.append(round(sales[sales.date == i].sales_rank.mean(),1))
    sales["salesrank"] = liste22

    sales = sales[["date","salesrank"]].dropna()
    sales.drop_duplicates(subset="date",ignore_index=True, inplace=True)

    sales.set_index("date", inplace=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.tight_layout()
    ax.plot(sales.index, sales["salesrank"])
    ax.set_xlabel('Datetime')
    ax.set_ylabel('Sales Rank')
    ax.set_title('Linear graph')
    #ax.set_xticklabels(sales.index, rotation=90)
    #fig.set_size_inches(3,2)

    sales.sort_index(inplace=True)
    datee = []
    for i in range(len(sales.index)):
        datee.append(datetime.datetime.now() - datetime.timedelta(days =i))
    sales["date2"] = datee[::-1]
    sales.date2 = sales.date2.dt.date.sort_values()
    sales.date2 = sales.date2.astype("datetime64[D]")
    sales = sales[["date2", "salesrank"]]
    sales = sales.set_index("date2")
    sales.sort_index(inplace = True)
    sales["month"] = sales.index.month


    s_mean = round(dict(sales.describe()[["salesrank"]])["salesrank"]["mean"],1)
    s_std = round(dict(sales.describe()[["salesrank"]])["salesrank"]["std"],1)
    s_min = round(dict(sales.describe()[["salesrank"]])["salesrank"]["min"],1)
    s_max = round(dict(sales.describe()[["salesrank"]])["salesrank"]["max"],1)
    s_25 = round(dict(sales.describe()[["salesrank"]])["salesrank"]["25%"],1)
    s_50 = round(dict(sales.describe()[["salesrank"]])["salesrank"]["50%"],1)
    s_75 = round(dict(sales.describe()[["salesrank"]])["salesrank"]["75%"],1)

    res = sm.tsa.seasonal_decompose(sales.salesrank).trend

    x = res.index
    y = res.values

    fft_srank_max = [x_val for x_val, y_val in zip(sales.month, res)if y_val ==pd.DataFrame(y).dropna().max().values[0]]
    fft_srank_min = [x_val for x_val, y_val in zip(sales.month, res)if y_val ==pd.DataFrame(y).dropna().min().values[0]]
    fft_srank_mean = round(pd.DataFrame(y).dropna().mean().values[0],2)
    fft_srank_std = round(pd.DataFrame(y).dropna().std().values[0],2)




    # Çizgi grafiğini oluşturmak için matplotlib kullanın
    fig2, ax = plt.subplots(figsize=(10, 7))
    fig2.tight_layout()
    
    ax.plot(x, y)
    #ax.set_xticklabels(x, rotation=90)

    # Eşik değeri belirleyin
    threshold = s_mean
    above_threshold_x = list(set([x_val for x_val, y_val in zip(sales.month, res) if y_val >= threshold]))

    # Eşik değeri üzerindeki alanları doldurun
    for i in range(len(y)):
        if y[i] <= threshold:
            ax.fill_between(x[i:i+2], y[i:i+2], threshold, color='red', alpha=0.5)

    ax.set_xlabel('Month')
    ax.set_ylabel('Sales Rank')
    ax.set_title('Trend Analysis')
    
    fig_t, ax = plt.subplots(figsize=(10, 7))
    ax.plot(x, y)

    # Türevin sıfır olduğu noktaları bulma
    dy = np.diff(y)
    tepe_noktalari = np.where(np.diff(np.sign(dy)) < 0)[0] + 1  # Tepe noktaları
    cukur_noktalari = np.where(np.diff(np.sign(dy)) > 0)[0] + 1  # Çukur noktaları

    # Tepe noktalarını grafiğe işaretleme
    ax.plot(x[tepe_noktalari], y[tepe_noktalari], 'ro', label='Tepe Noktaları')
    # Çukur noktalarını grafiğe işaretleme
    ax.plot(x[cukur_noktalari], y[cukur_noktalari], 'bo', label='Çukur Noktaları')

    ax.legend()

    ax.set_xlabel('Month')
    ax.set_ylabel('Sales Rank')
    ax.set_title('Trend Analysis 2')



    fig3, ax3 = plt.subplots(figsize=(10, 7))
    
    ax3.pie(bb_stats_df.groupby(["sellerType"])["percentageWon"].sum(), labels=bb_stats_df.groupby(["sellerType"])["percentageWon"].sum().index, autopct='%1.1f%%', startangle=30)
    center_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig3.gca().add_artist(center_circle)
    

    # Set the chart title
    ax3.set_title('SellerType PercentageWon')

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax3.axis('equal')

    # Display the donut chart using Streamlit
    
    
    parsed = {}
    try:
        if pd.isna(products[0]["stats_parsed"]) == False:
            prod = st.session_state.result_data.iloc[asin]["stats_parsed"]
            prod_keys = ['current', 'avg', 'avg30', 'avg90', 'avg180', 'avg365','min', 'max','outOfStockPercentage30', 'outOfStockPercentage90']
            pk_keys = 'AMAZON', 'NEW', 'SALES','LISTPRICE', 'WAREHOUSE', 'NEW_FBA', 'RATING', 'COUNT_REVIEWS'       
            for i in prod_keys:
                if (i != "min") & (i !="max"):
                    temp = []
                    for j in pk_keys:
                        try:
                            temp.append( prod[i][j])
                        except:
                            temp.append( None)
                    parsed[i] = temp
                else:
                    temp = []
                    for j in pk_keys:
                        try:
                            temp.append( prod[i][j][1])
                        except:
                            temp.append( None)
                    parsed[i] = temp    
        else:
            print("stats_parsed returns None!")
    except:
        pass




    ### with col1:

    with tab_x:
        st.header("         Details")    #predictions_proba = final_model.predict_proba(my_dict)
        st.title("Last Year Salesrank")
        col1, col2, col3 = st.columns(3)
        
       
        with col1:
            st.header("BSR")    #predictions_proba = final_model.predict_proba(my_dict)
            st.pyplot(fig,bbox_inches='tight', pad_inches=0.2)
           
            st.pyplot(fig3,bbox_inches='tight', pad_inches=0.2)
            
        with col2:
            st.header("Tranformed BSR")

            st.pyplot(fig2,bbox_inches='tight', pad_inches=0.2)
            st.pyplot(fig_t,bbox_inches='tight', pad_inches=0.2)


    with tab_y:
        st.write(pd.DataFrame(parsed, index = pk_keys))
        st.write(bb_stats_df)
        st.write("s_mean,s_std ,s_min ,s_max ,s_25,s_50,s_75")
        st.write(s_mean,s_std ,s_min ,s_max ,s_25,s_50,s_75)

        


       

        
        
    with tab_z:
        a = st.tabs([str(i) for i in range(len(imagesCSV))])
        for i in range(len(imagesCSV)):
            with a[i]:
                st.image(f"https://images-na.ssl-images-amazon.com/images/I/{imagesCSV[i]}")


        
           

  
else:
    st.write("Welcome!")



    # Add a file uploader widget
   
   
    # Check if a file was uploaded
    
        # Process the uploaded file
       
        #st.write(pd.DataFrame(contents))
    
                
    ### with col2: 
        # a = st.tabs([str(i) for i in range(len(imagesCSV))])
        # for i in range(len(imagesCSV)):
        #     with a[i]:
        #         st.image(f"https://images-na.ssl-images-amazon.com/images/I/{imagesCSV[i]}", width=300)

        




    
    
