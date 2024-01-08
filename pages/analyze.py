## import islemleri

import ast
import csv
import math
import time 
import keepa 
import base64
import datetime
import numpy as np
import pandas as pd 
import streamlit as st
from io import BytesIO
import statsmodels.api as sm
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter


accesskey2 = "apl5vvvmm4v4upvnnungrd6gfdtlkmb46jkrjj6toe06i03d7jndcahutsc3u43s"
accesskey = "c22mt3o8utp5jj1rnlobbptmc0gdfvjt3n6dg3p2gd2pi34ajj539vkrhluinu8s"
keepa_time = datetime.datetime(2011, 1 ,1)

product_keys = ["csv","stats","categories", "imagesCSV","manufacturer","title","lastUpdate","lastPriceChange","rootCategory", "productType","parentAsin","variationCSV","asin","domainId","type","brand","productGroup","partNumber","model","color","size","format","packageHeight","packageLength","packageWidth","packageWeight","packageQuantity","binding","numberOfItems","eanList","upcList","frequentlyBoughtTogether","features","description","promotions","coupon","availabilityAmazon","fbaFees","variations","itemHeight","itemLength","itemWidth","itemWeight","g","categoryTree","stats_parsed"]

# yaklasik moq degerlerinin tespiti
@st.cache_data 
def get_moq(param):
    moq = {}
    for i,t in param[["Coral Price","case"]].iterrows():
        ebob=math.gcd(round(t["Coral Price"]), int(t["case"]))
        ekok=(round(t["Coral Price"]) * int(t["case"]))/ebob 
        for p in range(1000,0,-1):
            if  400 <= ekok * p <= 550:
                if i not in moq.keys():
                    moq[i] = p
            
    print(param.shape)
    print(len(moq))
    return moq.values()

# ---------------------------------------------------------------------------------------


# Total urunlerden, satabilecegim urunlerin tespiti
@st.cache_data 
def match(df_200k, df_upc):
    match_upc1 = []
    match_upc2 = []
    upc_list = df_200k['Product Codes: UPC'].tolist()  # UPC listesini alın

    for j, p in df_upc.iterrows():
        for upc in upc_list:
            if upc is None:
                continue

            if str(p['UPC']) in str(upc):
                match_upc1.append(p)
                match_upc2.append(upc)

    # Match olan verileri içeren yeni bir DataFrame oluşturun
    df_match = pd.DataFrame({'Match_UPC1': match_upc1, 'Match_UPC2': match_upc2})
    return pd.concat([pd.DataFrame(match_upc1).reset_index(drop = True), pd.DataFrame(match_upc2, columns = ["Product Codes: UPC"]).reset_index(drop = True)], axis = 1)

# ---------------------------------------------------------------------------------------

# Uzun yoldan keepa urun taramasi. Hata alma orani yuksek!
@st.cache_data 
def send_req(param):
    with open('veriler.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(product_keys)  # Sütun başlıklarını yaz

        for i in range(len(param.ASIN)):
            api = keepa.Keepa(accesskey)
            current_token = api.tokens_left

            if current_token <= 10:
                time.sleep(3600)  # 60 dk beklesin
                print(current_token)

            try:
                products = api.query(param.ASIN[i], stats=30, buybox=1)
            except TimeoutError: 
                products = api.query(param.ASIN[i], stats=30, buybox=1)
            except ConnectionError: 
                products = api.query(param.ASIN[i], stats=30, buybox=1)
            except:
                products = {}

            temp = []

            for key in product_keys:
                try:
                    if key in products[0]:
                        temp.append(products[0][key])
                    else:
                        temp.append(None)
                except:
                    temp.append(None)

            writer.writerow(temp)  # Veriyi CSV dosyasına ekle

            print(current_token)
            time.sleep(0.5)
    file.close()

# ---------------------------------------------------------------------------------------

# "threading is for working in parallel, and async is for waiting in parallel".
@st.cache_data 
def send_req_threading(df_Wsku,stats):
    urls = df_Wsku.ASIN
    start = perf_counter()
    api = keepa.Keepa(accesskey)
    current_token = api.tokens_left
    print(current_token)

    def send_req(param):        
        try:
            products = api.query(param, stats=stats, buybox=1)
        except TimeoutError: 
            products = api.query(param, stats=stats, buybox=1)
        except ConnectionError: 
            products = api.query(param, stats=stats, buybox=1)
        except:
            products = {}
        
        temp = []

        for key in product_keys:
            try:
                if key in products[0]:
                    temp.append(products[0][key])
                else:
                    temp.append(None)
            except:
                temp.append(None)
        return temp
    
    api = keepa.Keepa(accesskey)
    current_token = api.tokens_left
    print(current_token)
    if current_token <= 10:
        time.sleep(3600)  # 60 dk beklesin
        print(current_token, "Token yukleniyor.")
    else:
        with ThreadPoolExecutor(60) as executor:
            results = list(executor.map(send_req, urls))
        

    stop = perf_counter()
    print("time taken:", stop - start)
    return results
# ---------------------------------------------------------------------------------------


# buybox price

def bb_price(param):
    if pd.isna(param):
       return "NO BB"
    elif  "buyBoxPrice" in param.keys():
        return round(param["buyBoxPrice"]/ 100, 2)
    else: 
        return None

            # ---------------------------------------------------------------------------------------

# buybox price 2

def bb_price_2(param):
    if pd.isna(param):
       return "NO BB"
    elif  "buyBoxPrice" in eval(param).keys():
        return  round((int(eval(param)["buyBoxPrice"]) / 100), 2)
    else: 
        return None
    

def get_bb_price(bb_price, df_Wsku):
    df_Wsku["bb_price"] = df_Wsku["stats_parsed"].apply(bb_price)   

# ---------------------------------------------------------------------------------------

# bb stats info

def bb_stats(param):
    bb_stats = []
    # total, amz, fba, fbm 
    bb_seller_number = ["0","0","0","0"]
    # We check the param value: None or Not None
    if pd.isna(param):
       return [None,None]
    else: 
        # We check the param value that which has "buyBoxStats" key or not.
        if "buyBoxStats" in param.keys():
            # Sometimes param value is not none. But it is falsy.
            if param["buyBoxStats"]:
                for t,j in param["buyBoxStats"].items():
                    datee = str((keepa_time + datetime.timedelta(minutes = j['lastSeen'])).year)+ "/" + str((keepa_time + datetime.timedelta(minutes = j['lastSeen'])).month) + "/" + str((keepa_time + datetime.timedelta(minutes = j['lastSeen'])).day)                  
                    if t == "ATVPDKIKX0DER":
                        bb_seller_number[1] = str(1)
                        bb_stats.append((str(t +" "+ "amz" +" "+ str(round(j["percentageWon"]))  +" "+  str(round((int(j["avgPrice"]) / 100),2)) +" "+ datee))) 
                    elif j["isFBA"]:
                        bb_seller_number[2] = str(int(bb_seller_number[2]) + 1)
                        bb_stats.append((str(t +" "+ "fba" +" "+ str(round(j["percentageWon"]))  +" "+  str(round((int(j["avgPrice"]) / 100),2)) +" "+ datee)))
                    else:
                        bb_seller_number[3] = str(int(bb_seller_number[3]) + 1)
                        bb_stats.append((str(t +" "+ "fbm" +" "+ str(round(j["percentageWon"]))  +" "+  str(round((int(j["avgPrice"]) / 100),2)) +" "+ datee)))
                    bb_seller_number[0] = str(int(bb_seller_number[1]) + int(bb_seller_number[2]) + int(bb_seller_number[3]))
                return [bb_stats, bb_seller_number]
            else: return [None,None] # no bb no seller info.
        else: return [None,None]
        
            # ---------------------------------------------------------------------------------------

# bb stats info 2

def bb_stats_2(param):
    bb_stats = []
    # total, amz, fba, fbm 
    bb_seller_number = ["0","0","0","0"]
    # We check the param value: None or Not None
    if pd.isna(param):
       return [None,None]
    else: 
        # We check the param value that which has "buyBoxStats" key or not.
        if "buyBoxStats" in eval(param).keys():
            # Sometimes param value is not none. But it is falsy.
            if eval(param)["buyBoxStats"]:
                for t,j in eval(param)["buyBoxStats"].items():
                    datee = str((keepa_time + datetime.timedelta(minutes = j['lastSeen'])).year)+ "/" + str((keepa_time + datetime.timedelta(minutes = j['lastSeen'])).month) + "/" + str((keepa_time + datetime.timedelta(minutes = j['lastSeen'])).day)                  
                    if t == "ATVPDKIKX0DER":
                        bb_seller_number[1] = str(1)
                        bb_stats.append((str(t +" "+ "amz" +" "+ str(round(j["percentageWon"]))  +" "+  str(round((int(j["avgPrice"]) / 100),2)) +" "+ datee))) 
                    elif j["isFBA"]:
                        bb_seller_number[2] = str(int(bb_seller_number[2]) + 1)
                        bb_stats.append((str(t +" "+ "fba" +" "+ str(round(j["percentageWon"]))  +" "+  str(round((int(j["avgPrice"]) / 100),2)) +" "+ datee)))
                    else:
                        bb_seller_number[3] = str(int(bb_seller_number[3]) + 1)
                        bb_stats.append((str(t +" "+ "fbm" +" "+ str(round(j["percentageWon"]))  +" "+  str(round((int(j["avgPrice"]) / 100),2)) +" "+ datee)))
                    bb_seller_number[0] = str(int(bb_seller_number[1]) + int(bb_seller_number[2]) + int(bb_seller_number[3]))
                return [bb_stats, bb_seller_number]
            else: return [None,None] # no bb no seller info.
        else: return [None,None]



def get_bb_stats(bb_stats, df_Wsku, day, stats):
    features = [day + "_bb_stats",day + "_bb_seller_number"]
    print(features)
    for i in range(len(features)):
        df_Wsku[features[i]] = df_Wsku[stats].apply(bb_stats).apply(lambda x: x[i])

# ---------------------------------------------------------------------------------------

# best season info
@st.cache_data 
def product_time(min_list):
    """Convert raw keepa time to datetime!"""
    keepa_time = datetime.datetime(2011, 1 ,1)
    return [keepa_time + datetime.timedelta(minutes = i) for i in min_list]
@st.cache_data 
def sales_analyze(param):
    try:   
        sales = pd.DataFrame()
        sales["datetime"] = product_time(param)[3][::-1][1::2]
        sales["sales_rank"] = param[3][::-1][::2]
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
        # s_std = round(dict(sales.describe()[["salesrank"]])["salesrank"]["std"],1)
        # s_min = round(dict(sales.describe()[["salesrank"]])["salesrank"]["min"],1)
        # s_max = round(dict(sales.describe()[["salesrank"]])["salesrank"]["max"],1)
        # s_25 = round(dict(sales.describe()[["salesrank"]])["salesrank"]["25%"],1)
        # s_50 = round(dict(sales.describe()[["salesrank"]])["salesrank"]["50%"],1)
        # s_75 = round(dict(sales.describe()[["salesrank"]])["salesrank"]["75%"],1)

        res = sm.tsa.seasonal_decompose(sales.salesrank).trend
        x = res.index
        y = res.values

        # fft_srank_max = [x_val for x_val, y_val in zip(sales.month, res)if y_val ==pd.DataFrame(y).dropna().max().values[0]]
        # fft_srank_min = [x_val for x_val, y_val in zip(sales.month, res)if y_val ==pd.DataFrame(y).dropna().min().values[0]]
        # fft_srank_mean = round(pd.DataFrame(y).dropna().mean().values[0],2)
        # fft_srank_std = round(pd.DataFrame(y).dropna().std().values[0],2)

        # Eşik değeri belirleyin
        threshold = s_mean
        above_threshold_x = list(set([x_val for x_val, y_val in zip(sales.month, res) if y_val >= threshold]))
        # Türevin sıfır olduğu noktaları bulma
        dy = np.diff(y)
        tepe_noktalari = np.where(np.diff(np.sign(dy)) < 0)[0] + 1  # Tepe noktaları
        cukur_noktalari = np.where(np.diff(np.sign(dy)) > 0)[0] + 1  # Çukur noktaları
        return above_threshold_x

    except:
        return None
    
            # ---------------------------------------------------------------------------------------
@st.cache_data 
def sales_analyze_2(param):
    try:   
        sales = pd.DataFrame()
        sales["datetime"] = product_time(ast.literal_eval(param))[3][::-1][1::2]
        sales["sales_rank"] = ast.literal_eval(param)[3][::-1][::2]
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
        # s_std = round(dict(sales.describe()[["salesrank"]])["salesrank"]["std"],1)
        # s_min = round(dict(sales.describe()[["salesrank"]])["salesrank"]["min"],1)
        # s_max = round(dict(sales.describe()[["salesrank"]])["salesrank"]["max"],1)
        # s_25 = round(dict(sales.describe()[["salesrank"]])["salesrank"]["25%"],1)
        # s_50 = round(dict(sales.describe()[["salesrank"]])["salesrank"]["50%"],1)
        # s_75 = round(dict(sales.describe()[["salesrank"]])["salesrank"]["75%"],1)

        res = sm.tsa.seasonal_decompose(sales.salesrank).trend
        x = res.index
        y = res.values

        # fft_srank_max = [x_val for x_val, y_val in zip(sales.month, res)if y_val ==pd.DataFrame(y).dropna().max().values[0]]
        # fft_srank_min = [x_val for x_val, y_val in zip(sales.month, res)if y_val ==pd.DataFrame(y).dropna().min().values[0]]
        # fft_srank_mean = round(pd.DataFrame(y).dropna().mean().values[0],2)
        # fft_srank_std = round(pd.DataFrame(y).dropna().std().values[0],2)

        # Eşik değeri belirleyin
        threshold = s_mean
        above_threshold_x = list(set([x_val for x_val, y_val in zip(sales.month, res) if y_val >= threshold]))
        # Türevin sıfır olduğu noktaları bulma
        dy = np.diff(y)
        tepe_noktalari = np.where(np.diff(np.sign(dy)) < 0)[0] + 1  # Tepe noktaları
        cukur_noktalari = np.where(np.diff(np.sign(dy)) > 0)[0] + 1  # Çukur noktaları
        return above_threshold_x

    except:
        return None
    
# ---------------------------------------------------------------------------------------

# dimension info

def get_dim():
    st.session_state.result_data["packageHeight"]= st.session_state.result_data["packageHeight"].apply(lambda x: round(x *  0.0393701, 2))
    st.session_state.result_data["packageLength"]= st.session_state.result_data["packageLength"].apply(lambda x: round(x * 0.0393701, 2))
    st.session_state.result_data["packageWidth"]= st.session_state.result_data["packageWidth"].apply(lambda x: round(x * 0.0393701,2))
    st.session_state.result_data["packageWeight2"]= st.session_state.result_data["packageWeight"].apply(lambda x: round(x  *  0.035274,2)) 
    st.session_state.result_data["packageWeight"]= st.session_state.result_data["packageWeight"].apply(lambda x: round(x *  0.00220462,2))
    st.session_state.result_data["minn"] = [j.sort_values()[0] if pd.isna(j).any() == False else None for i,j in st.session_state.result_data[["packageHeight","packageWidth", "packageLength"]].iterrows()]
    st.session_state.result_data["mid"] = [j.sort_values()[1] if pd.isna(j).any() == False else None for i,j in st.session_state.result_data[["packageHeight","packageWidth", "packageLength"]].iterrows()]
    st.session_state.result_data["maxx"] = [j.sort_values()[2] if pd.isna(j).any() == False else None  for i,j in st.session_state.result_data[["packageHeight","packageWidth", "packageLength"]].iterrows()]

# ---------------------------------------------------------------------------------------

def low_price(param,prod_type):  
    # https://www.junglescout.com/blog/amazon-fba-fees/
    # # prod type i otomatik olacak şekilde ayarla.
    FBA_fee = 0
    low_price = False


    if datetime.datetime.now().month <= 9:
        storage_std = round(param["maxx"]*0.0833333 * param["minn"]*0.0833333 * param["mid"]*0.0833333 * 0.87,2)    
        storage_ovs = round(param["maxx"]*0.0833333 * param["minn"]*0.0833333 * param["mid"]*0.0833333 * 0.56,2)  
    else:
        storage_std = round(param["maxx"]*0.0833333 * param["minn"]*0.0833333 * param["mid"]*0.0833333 * 2.40,2)    
        storage_ovs = round(param["maxx"]*0.0833333 * param["minn"]*0.0833333 * param["mid"]*0.0833333 * 1.40,2)  
    

    unit_F_cost = round(param["maxx"]*0.0254 * param["minn"]*0.0254 * param["mid"]*0.0254 * 400,2)


    if pd.isna(param['bb_price']):
       return [None,None,None, None, None, None, unit_F_cost, storage_std, storage_ovs, low_price]

    elif param['bb_price'] == "NO BB":
       return [None,None,None, None, None, None, unit_F_cost, storage_std, storage_ovs, low_price]
    
    elif param["bb_price"] <= 10: # low price fba fee for <= 10 
        if (param["maxx"] <= 15) & (param["minn"] <= 0.75) & (param["mid"] <= 12) & (param["packageWeight2"] <= 16):
            low_price = True
            if param["packageWeight2"] <= 4:
                FBA_fee = 2.45
            elif (param["packageWeight2"] > 4) & (param["packageWeight2"] <= 8):
                FBA_fee = 2.63
            elif (param["packageWeight2"] > 8) & (param["packageWeight2"] <= 12):
                FBA_fee = 2.81
            elif (param["packageWeight2"] > 12) & (param["packageWeight2"] <= 16):
                FBA_fee = 3.0
            else:
                print("small standart degil")
               
        elif (param["maxx"] <= 18) & (param["minn"] <= 8) & (param["mid"] <= 14) :
            low_price = True
            if param["packageWeight2"] <= 4:
                FBA_fee = 3.09
            elif (param["packageWeight2"] > 4) & (param["packageWeight2"] <= 8):
                FBA_fee = 3.31
            elif (param["packageWeight2"] > 8) & (param["packageWeight2"] <= 12):
                FBA_fee = 3.47
            elif (param["packageWeight2"] > 12) & (param["packageWeight2"] <= 16):
                FBA_fee = 3.98
            elif (param["packageWeight2"] > 16) & (param["packageWeight"] <= 1.5):
                FBA_fee = 4.63
            elif (param["packageWeight"] > 1.5) & (param["packageWeight"] <= 2):
                FBA_fee = 4.92
            elif (param["packageWeight"] > 2) & (param["packageWeight"] <= 2.5):
                FBA_fee = 5.33
            elif (param["packageWeight"] > 2.5) & (param["packageWeight"] <= 3):
                FBA_fee = 5.62
            else:
                rounded_weight = math.ceil(param["packageWeight"] * 2) / 2  # Ağırlığı yukarıya yuvarla (örneğin 4.32 => 4.5)
                additional_weight = rounded_weight - 3
                FBA_fee = round(6.40 + (additional_weight * 0.32),2)
    
        elif (param["maxx"] <= 60) & (param["mid"] <= 30) & (param["packageWeight"] > 20) & (param["packageWeight"] <= 70) & ((((param["minn"] + param["mid"])*2)+param["maxx"]) <= 130):
            low_price = True
            rounded_weight = math.ceil(param["packageWeight"] * 2) / 2  # Ağırlığı yukarıya yuvarla (örneğin 4.32 => 4.5)
            additional_weight = rounded_weight - 3
            FBA_fee = round(8.96 + (additional_weight * 0.42),2)
        
        elif (param["maxx"] <= 108) & (param["packageWeight"] > 70) & (param["packageWeight"] <= 150) & ((((param["minn"] + param["mid"])*2)+param["maxx"]) <= 130):
            low_price = True
            rounded_weight = math.ceil(param["packageWeight"] * 2) / 2  # Ağırlığı yukarıya yuvarla (örneğin 4.32 => 4.5)
            additional_weight = rounded_weight - 3
            FBA_fee = round(18.28 + (additional_weight * 0.42),2)
                
        #elif (param["maxx"] <= 108) & (param["packageWeight"] > 70) & (param["packageWeight"] <= 150) & ((((param["minn"] + param["mid"])*2)+param["maxx"]) <= 165):
        #    rounded_weight = math.ceil(param["packageWeight"] * 2) / 2  # Ağırlığı yukarıya yuvarla (örneğin 4.32 => 4.5)
        #    additional_weight = rounded_weight - 3
        #    FBA_fee = 89.21 + (additional_weight * 0.83)
        
        #elif (param["maxx"] > 108) & (param["packageWeight"] > 150) & ((((param["minn"] + param["mid"])*2)+param["maxx"]) > 165):
        #    rounded_weight = math.ceil(param["packageWeight"] * 2) / 2  # Ağırlığı yukarıya yuvarla (örneğin 4.32 => 4.5)
        #    additional_weight = rounded_weight - 3
        #    FBA_fee = 89.21 + (additional_weight * 0.83)
        else:
            FBA_fee = 999999999
    
    
    # normal fba fee for price > 10 
    
    else:
        #print("price küçük evet")
        if (param["maxx"] <= 15) & (param["minn"] <= 0.75) & (param["mid"] <= 12) & (param["packageWeight2"] <= 16):
            if param["packageWeight2"] <= 4:
                FBA_fee = 3.22
            elif (param["packageWeight2"] > 4) & (param["packageWeight2"] <= 8):
                FBA_fee = 3.40
            elif (param["packageWeight2"] > 8) & (param["packageWeight2"] <= 12):
                FBA_fee = 3.58
            elif (param["packageWeight2"] > 12) & (param["packageWeight2"] <= 16):
                FBA_fee = 3.77
            else:
                print("small standart degil")
               
        elif (param["maxx"] <= 18) & (param["minn"] <= 8) & (param["mid"] <= 14) :
            if param["packageWeight2"] <= 4:
                FBA_fee = 3.86
            elif (param["packageWeight2"] > 4) & (param["packageWeight2"] <= 8):
                FBA_fee = 4.08
            elif (param["packageWeight2"] > 8) & (param["packageWeight2"] <= 12):
                FBA_fee = 4.24
            elif (param["packageWeight2"] > 12) & (param["packageWeight2"] <= 16):
                FBA_fee = 4.75
            elif (param["packageWeight2"] > 16) & (param["packageWeight"] <= 1.5):
                FBA_fee = 5.40
            elif (param["packageWeight"] > 1.5) & (param["packageWeight"] <= 2):
                FBA_fee = 5.69
            elif (param["packageWeight"] > 2) & (param["packageWeight"] <= 2.5):
                FBA_fee = 6.10
            elif (param["packageWeight"] > 2.5) & (param["packageWeight"] <= 3):
                FBA_fee = 6.39
            else:
                rounded_weight = math.ceil(param["packageWeight"] * 2) / 2  # Ağırlığı yukarıya yuvarla (örneğin 4.32 => 4.5)
                additional_weight = rounded_weight - 3
                FBA_fee = round(7.17 + (additional_weight * 0.32),2)
                
        elif (param["maxx"] <= 60) & (param["mid"] <= 30) & (param["packageWeight"] > 20) & (param["packageWeight"] <= 70) & ((((param["minn"] + param["mid"])*2)+param["maxx"]) <= 130):
            rounded_weight = math.ceil(param["packageWeight"] * 2) / 2  # Ağırlığı yukarıya yuvarla (örneğin 4.32 => 4.5)
            additional_weight = rounded_weight - 3
            FBA_fee = round(9.73 + (additional_weight * 0.42),2)
            
        elif (param["maxx"] <= 108) & (param["packageWeight"] > 70) & (param["packageWeight"] <= 150) & ((((param["minn"] + param["mid"])*2)+param["maxx"]) <= 130):
            rounded_weight = math.ceil(param["packageWeight"] * 2) / 2  # Ağırlığı yukarıya yuvarla (örneğin 4.32 => 4.5)
            additional_weight = rounded_weight - 3
            FBA_fee = round(19.05 + (additional_weight * 0.42),2)
                
        #elif (param["maxx"] <= 108) & (param["packageWeight"] > 70) & (param["packageWeight"] <= 150) & ((((param["minn"] + param["mid"])*2)+param["maxx"]) <= 165):
        #    rounded_weight = math.ceil(param["packageWeight"] * 2) / 2  # Ağırlığı yukarıya yuvarla (örneğin 4.32 => 4.5)
        #    additional_weight = rounded_weight - 3
        #    FBA_fee = 89.21 + (additional_weight * 0.83)
        
        #elif (param["maxx"] > 108) & (param["packageWeight"] > 150) & ((((param["minn"] + param["mid"])*2)+param["maxx"]) > 165):
        #    rounded_weight = math.ceil(param["packageWeight"] * 2) / 2  # Ağırlığı yukarıya yuvarla (örneğin 4.32 => 4.5)
        #    additional_weight = rounded_weight - 3
        #    FBA_fee = 158.49 + (additional_weight * 0.83)
        else:
            FBA_fee = 999999999
    
      
    #print("storage",storage_std, storage_ovs)
    #print("unit_F_cost",unit_F_cost)
    #print("FBA_fee",FBA_fee)

    if prod_type == "grocery":
        if param["bb_price"] <= 15:
            referral_fee = round((param["bb_price"] * 8 / 100),2)
        else:
            referral_fee = round((param["bb_price"] * 15/100),2)

    elif prod_type == "office":
        referral_fee = round((param["bb_price"] * 15/100),2)
        if referral_fee < 0.3:
            referral_fee = 0.3
    #
    # 
    # add different categories!
    # 
    # #
    else: pass

    #print("referral_fee",referral_fee)

    if param["packageWeight"] <= 20:
        total_fee = FBA_fee + referral_fee +unit_F_cost + storage_std
    else:
        total_fee = FBA_fee + referral_fee +unit_F_cost + storage_ovs

    cost =  param["Coral Price"]
    profit = round(param["bb_price"] - total_fee-cost,2)
    #print("profit",profit)
    margin = round(profit/param["bb_price"]*100,2)
    #print("margin",margin)
    roi = round(profit/(cost+unit_F_cost)*100,2)
    return [roi, margin, profit, total_fee, FBA_fee, referral_fee, unit_F_cost, storage_std, storage_ovs, low_price]
    

def get_roi(df_Wsku, prod_type):
    features = ["roi", "margin", "profit", "total_fee", "FBA_fee", "referral_fee", "unit_F_cost", "storage_std", "storage_ovs","low_price"]
    for i in range(0,len(features)):
        df_Wsku[features[i]] = [low_price(k,prod_type)[i] for p,k in df_Wsku[['bb_price','minn', 'mid', 'maxx', 'packageHeight', 'packageLength', 'packageWidth', 'packageWeight', 'packageWeight2',"Coral Price"]].iterrows()]

# ---------------------------------------------------------------------------------------

# oos info

def oos(param):
    par = ["outOfStockPercentage30","outOfStockPercentage90"]
    result = [None, None]
    if pd.isna(param) == False:
        for i in range(0,len(par)):
            if par[i] in param.keys():
                if "AMAZON" in param[par[i]]:
                    result[i] = round(param[par[i]]["AMAZON"]*100,2)
        return result
    else: return result

            # ---------------------------------------------------------------------------------------

# oos info 2

def oos_2(param):
    par = ["outOfStockPercentage30","outOfStockPercentage90"]
    result = [None, None]
    if pd.isna(param) == False:
        for i in range(0,len(par)):
            if par[i] in eval(param).keys():
                if "AMAZON" in eval(param)[par[i]]:
                    result[i] = round(eval(param)[par[i]]["AMAZON"]*100,2)
        return result
    else: return result


def get_oos(oos, df_Wsku, day, stats):
    features = [day + "_oos30",day + "_oos90"]
    print(features)
    for i in range(len(features)):
        df_Wsku[features[i]] = df_Wsku[stats].apply(oos).apply(lambda x: x[i])

# ---------------------------------------------------------------------------------------

# bsr info

def bsr(param):
    par = ["current","avg30","avg90"]
    result = [None, None, None]
    if pd.isna(param) == False:
        for i in range(0,len(par)):
            if par[i] in param.keys(): 
                if "SALES" in param[par[i]]:
                    result[i] = param[par[i]]["SALES"] 
        return result
    else: return result

            # ---------------------------------------------------------------------------------------

# bsr info 2
 
def bsr_2(param):
    par = ["current","avg30","avg90"]
    result = [None, None, None]
    if pd.isna(param) == False:
        for i in range(0,len(par)):
            if par[i] in eval(param).keys(): 
                if "SALES" in eval(param)[par[i]]:
                    result[i] = eval(param)[par[i]]["SALES"] 
        return result
    else: return result

def get_bsr(bsr, df_Wsku, stats):
    features = ["bsr_c","bsr_30","bsr_90"]
    for i in range(len(features)):
        df_Wsku[features[i]] = df_Wsku[stats].apply(bsr).apply(lambda x: x[i])
# --------------------


@st.cache_data
def process_uploaded_file(file):
    if file is not None:
        try:
            if file.name.endswith(".csv"):
                if file.name.startswith("csvproductexport"):
                    df = pd.read_csv(file, dtype={"SKU": str, "UPC": str}, sep = ";").copy()
                else:
                    df = pd.read_csv(file, dtype={"SKU": str, "UPC": str}).copy()
            elif file.name.endswith(".xlsx"):
                df = pd.read_excel(file, dtype={"SKU": str, "UPC": str}).copy()
            else:
                st.write("Please upload a file with the right format!!!")
                return None
            return df
        except Exception as e:
            st.write(f"Error reading the file: {e}")
            return None



st.write("get_data dan gelen veri:", st.session_state.result_data)

if not st.session_state.result_data.empty:
    

    for i in ["stats_parsed","stats_parsed_90","stats_parsed_180","stats_parsed_360"]:
        if i in st.session_state.result_data.columns:
            get_dim()
            st.write(st.session_state.result_data)
            get_bsr(bsr, st.session_state.result_data, i)
            get_bb_price(bb_price, st.session_state.result_data)
            get_bb_stats(bb_stats,st.session_state.result_data,i,i)
            get_oos(oos,st.session_state.result_data, i,i)
        else: 
            pass
            





st.write(st.session_state.result_data)





st.title('Counter Example')
if 'temp_data' not in st.session_state:
    st.session_state.temp_data = pd.DataFrame()

upload_main = st.file_uploader("Upload Main Files!")
df_uploaded = process_uploaded_file(upload_main)
st.session_state.temp_data = pd.DataFrame()


#st.write(df_uploaded)
col1, col2,col3,col4 = st.columns(4)
with col1: option1 = st.checkbox("30")
with col2: option2 = st.checkbox("90")
with col3: option3 = st.checkbox("180")
with col4: option4 = st.checkbox("360")

selected_options = [i for i in [option1, option2, option3, option4] if i]
#temp_data = pd.DataFrame()

for i in range(len(selected_options)):
    if selected_options[i]:
        choice = st.number_input(f"Pick a number_{i}", 0, 100)
        if choice:
            uploaded_files = st.file_uploader(f"Choose files (csv or xlsx)_{i}", accept_multiple_files=True)
            data = pd.DataFrame()
            if uploaded_files:
                for j in range(choice):
                    #if str(data_stats) in uploaded_files[j].name:
                    if i != 0:
                        data = pd.concat([data,process_uploaded_file(uploaded_files[j])[["stats_parsed"]]], ignore_index=True)
                    else: 
                        data = pd.concat([data,process_uploaded_file(uploaded_files[j])], ignore_index=True)
            if i != 0:
                data.columns = [k+ f"_{i}" for k in data.columns]
            
             
            st.session_state.temp_data = pd.concat([st.session_state.temp_data,data],axis = 1)
#st.write(set(df_uploaded.columns).intersection(set(temp_data.columns)))
st.session_state.temp_data = pd.concat([df_uploaded, st.session_state.temp_data], axis = 1)
st.write(st.session_state.temp_data)
if not st.session_state.temp_data.empty:
    get_dim()
    st.write(st.session_state.temp_data)
    get_bsr(bsr_2, st.session_state.temp_data, "stats_parsed")
    get_bb_price(bb_price_2, st.session_state.temp_data)
    get_bb_stats(bb_stats_2,st.session_state.temp_data,"","stats_parsed")
    get_oos(oos_2,st.session_state.temp_data, "","stats_parsed")



    for i in range(len(selected_options)):
        if f"stats_parsed_{i}" in st.session_state.temp_data.columns:
            get_bb_stats(bb_stats_2,st.session_state.temp_data, i ,f"stats_parsed_{i}")
            get_oos(oos_2,st.session_state.temp_data, i,f"stats_parsed_{i}")
        


    selected_option_2 = st.selectbox("Lütfen roi  ürün kategorisini secin!:", ["office","grocery"])
    get_roi(st.session_state.temp_data, selected_option_2)

    st.session_state.temp_data

    output_0 = BytesIO()
    st.session_state.temp_data.to_excel(output_0, index=False, engine='openpyxl')
    output_0.seek(0)

    # İndirme bağlantısını göster
    b64 = base64.b64encode(output_0.read()).decode()
    st.markdown(f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="final_file.xlsx">Download Final File!</a>', unsafe_allow_html=True)


    st.success('Analysis is complete!', icon="✅")

else: st.error("There is no data!!")


st.session_state.temp_data["checkbox"] = False


st.data_editor(
    st.session_state.temp_data,
    
    disabled=[item for item in st.session_state.temp_data.columns if item != "checkbox"],

    hide_index=True,
)


# @st.cache_data
# def load_data(param):
#     return param.copy()

# load_data(temp_data)
## get seasonality yi de göster. 
## güzel bi görsel arayüz oluştur.


#data_files = st.radio("Lütfen bir seçenek seçin:", [1,2,3,4])

# choice_2 = st.number_input("Pick a number_2", 0, 100)
# data_stats_2 = st.selectbox("aa",[30, 90, 180, 360])

# uploaded_files_2 = st.file_uploader("Choose files (csv or xlsx)!", accept_multiple_files=True)
# data_2 = pd.DataFrame()
# if uploaded_files_2:
#     for i in range(choice_2):
#         if str(data_stats_2) in uploaded_files_2[i].name:
#             data = pd.concat([data,process_uploaded_file(uploaded_files_2[i])], ignore_index=True)
# # st.write(data_2)



# # dataframes = []

# # for i in range(choice):
# #     # Her bir DataFrame'i oluşturun, burada verileri dilediğiniz gibi ayarlayabilirsiniz.
# #     data = {
# #         'Sutun1': [1, 2, 3],
# #         'Sutun2': ['A', 'B', 'C']
# #     }
    
# #     df = pd.DataFrame(data)
    
# #     # DataFrame'i bir listeye ekleyin
# #     dataframes.append(df)

# # # Oluşturulan DataFrame'leri göstermek için döngü kullanabilirsiniz
# # for i, df in enumerate(dataframes):
# #     st.write(f"DataFrame {i+1}:")
# #     st.write(df)




# selected_options = [option for option, selected in zip(["Seçenek 1", "Seçenek 2", "Seçenek 3"], [option1, option2, option3]) if selected]

# st.write("Seçilen Seçenekler:", selected_options)

# selected_options = [selected for n  [option1, option2, option3] if selected]
target =['Locale', 'Image', 'Title', 'Description & Features: Description', 'Sales Rank: Current', 'Sales Rank: 30 days avg.', 'Sales Rank: 90 days avg.', 'Sales Rank: 180 days avg.', 'Sales Rank: 1 day drop %', 'Sales Rank: 7 days drop %', 'Sales Rank: 30 days drop %', 'Sales Rank: 90 days drop %', 'Sales Rank: Drop since last visit', 'Sales Rank: Drop % since last visit', 'Sales Rank: Last visit', 'Sales Rank: Lowest', 'Sales Rank: Highest', 'Sales Rank: Drops last 30 days', 'Sales Rank: Drops last 90 days', 'Sales Rank: Drops last 180 days', 'Sales Rank: Reference', 'Sales Rank: Subcategory Sales Ranks', 'Bought in past month', 'Reviews: Rating', 'Reviews: Review Count', 'Reviews: Review Count - 30 days avg.', 'Reviews: Review Count - 90 days avg.', 'Reviews: Review Count - 180 days avg.', 'Reviews: Review Count - 1 day drop %', 'Reviews: Review Count - 7 days drop %', 'Reviews: Review Count - 30 days drop %', 'Reviews: Review Count - 90 days drop %', 'Last Price Change', 'Last Update', 'Last Offer Update', 'Buy Box: Current', 'Buy Box: 30 days avg.', 'Buy Box: 90 days avg.', 'Buy Box: 180 days avg.', 'Buy Box: 1 day drop %', 'Buy Box: 7 days drop %', 'Buy Box: 30 days drop %', 'Buy Box: 90 days drop %', 'Buy Box: Drop since last visit', 'Buy Box: Drop % since last visit', 'Buy Box: Last visit', 'Buy Box: Lowest', 'Buy Box: Highest', 'Buy Box: Stock', 'Buy Box out of stock percentage: 90 days OOS %', 'Buy Box Seller', 'Buy Box: Is FBA', 'Buy Box: Unqualified', 'Buy Box: Preorder', 'Buy Box: Backorder', 'Buy Box: Prime Exclusive', 'Amazon: Current', 'Amazon: 30 days avg.', 'Amazon: 90 days avg.', 'Amazon: 180 days avg.', 'Amazon: 1 day drop %', 'Amazon: 7 days drop %', 'Amazon: 30 days drop %', 'Amazon: 90 days drop %', 'Amazon: Last visit', 'Amazon: Drop % since last visit', 'Amazon: Drop since last visit', 'Amazon: Lowest', 'Amazon: Highest', 'Amazon: Stock', 'Amazon out of stock percentage: 90 days OOS %', 'Amazon: Availability of the Amazon offer', 'Amazon: Amazon offer shipping delay', 'New: Current', 'New: 30 days avg.', 'New: 90 days avg.', 'New: 180 days avg.', 'New: 1 day drop %', 'New: 7 days drop %', 'New: 30 days drop %', 'New: 90 days drop %', 'New: Drop since last visit', 'New: Drop % since last visit', 'New: Last visit', 'New: Lowest', 'New: Highest', 'New out of stock percentage: 90 days OOS %', 'MAP restriction', 'New, 3rd Party FBA: Current', 'New, 3rd Party FBA: 30 days avg.', 'New, 3rd Party FBA: 90 days avg.', 'New, 3rd Party FBA: 180 days avg.', 'New, 3rd Party FBA: 1 day drop %', 'New, 3rd Party FBA: 7 days drop %', 'New, 3rd Party FBA: 30 days drop %', 'New, 3rd Party FBA: 90 days drop %', 'New, 3rd Party FBA: Drop since last visit', 'New, 3rd Party FBA: Drop % since last visit', 'New, 3rd Party FBA: Last visit', 'New, 3rd Party FBA: Lowest', 'New, 3rd Party FBA: Highest', 'Lowest FBA Seller', 'FBA Fees:', 'Referral Fee %', 'Referral Fee based on current Buy Box price', 'New, 3rd Party FBM: Current', 'New, 3rd Party FBM: 30 days avg.', 'New, 3rd Party FBM: 90 days avg.', 'New, 3rd Party FBM: 180 days avg.', 'New, 3rd Party FBM: 1 day drop %', 'New, 3rd Party FBM: 7 days drop %', 'New, 3rd Party FBM: 30 days drop %', 'New, 3rd Party FBM: 90 days drop %', 'New, 3rd Party FBM: Drop since last visit', 'New, 3rd Party FBM: Drop % since last visit', 'New, 3rd Party FBM: Last visit', 'New, 3rd Party FBM: Lowest', 'New, 3rd Party FBM: Highest', 'Lowest FBM Seller', 'New, Prime Exclusive: Current', 'New, Prime Exclusive: 30 days avg.', 'New, Prime Exclusive: 90 days avg.', 'New, Prime Exclusive: 180 days avg.', 'New, Prime Exclusive: 1 day drop %', 'New, Prime Exclusive: 7 days drop %', 'New, Prime Exclusive: 30 days drop %', 'New, Prime Exclusive: 90 days drop %', 'New, Prime Exclusive: Drop since last visit', 'New, Prime Exclusive: Drop % since last visit', 'New, Prime Exclusive: Last visit', 'New, Prime Exclusive: Lowest', 'New, Prime Exclusive: Highest', 'Lightning Deals: Current', 'Lightning Deals: Upcoming Deal', 'Buy Box Used: Current', 'Buy Box Used: 30 days avg.', 'Buy Box Used: 90 days avg.', 'Buy Box Used: 180 days avg.', 'Buy Box Used: 1 day drop %', 'Buy Box Used: 7 days drop %', 'Buy Box Used: 30 days drop %', 'Buy Box Used: 90 days drop %', 'Buy Box Used: Drop since last visit', 'Buy Box Used: Drop % since last visit', 'Buy Box Used: Last visit', 'Buy Box Used: Lowest', 'Buy Box Used: Highest', 'Buy Box Used: Seller', 'Buy Box Used: Is FBA', 'Buy Box Used: Condition', 'Used: Current', 'Used: 30 days avg.', 'Used: 90 days avg.', 'Used: 180 days avg.', 'Used: 1 day drop %', 'Used: 7 days drop %', 'Used: 30 days drop %', 'Used: 90 days drop %', 'Used: Drop since last visit', 'Used: Drop % since last visit', 'Used: Last visit', 'Used: Lowest', 'Used: Highest', 'Used out of stock percentage: 90 days OOS %', 'Used, like new: Current', 'Used, like new: 30 days avg.', 'Used, like new: 90 days avg.', 'Used, like new: 180 days avg.', 'Used, like new: 1 day drop %', 'Used, like new: 7 days drop %', 'Used, like new: 30 days drop %', 'Used, like new: 90 days drop %', 'Used, like new: Drop since last visit', 'Used, like new: Drop % since last visit', 'Used, like new: Last visit', 'Used, like new: Lowest', 'Used, like new: Highest', 'Used, very good: Current', 'Used, very good: 30 days avg.', 'Used, very good: 90 days avg.', 'Used, very good: 180 days avg.', 'Used, very good: 1 day drop %', 'Used, very good: 7 days drop %', 'Used, very good: 30 days drop %', 'Used, very good: 90 days drop %', 'Used, very good: Drop since last visit', 'Used, very good: Drop % since last visit', 'Used, very good: Last visit', 'Used, very good: Lowest', 'Used, very good: Highest', 'Used, good: Current', 'Used, good: 30 days avg.', 'Used, good: 90 days avg.', 'Used, good: 180 days avg.', 'Used, good: 1 day drop %', 'Used, good: 7 days drop %', 'Used, good: 30 days drop %', 'Used, good: 90 days drop %', 'Used, good: Drop since last visit', 'Used, good: Drop % since last visit', 'Used, good: Last visit', 'Used, good: Lowest', 'Used, good: Highest', 'Used, acceptable: Current', 'Used, acceptable: 30 days avg.', 'Used, acceptable: 90 days avg.', 'Used, acceptable: 180 days avg.', 'Used, acceptable: 1 day drop %', 'Used, acceptable: 7 days drop %', 'Used, acceptable: 30 days drop %', 'Used, acceptable: 90 days drop %', 'Used, acceptable: Drop since last visit', 'Used, acceptable: Drop % since last visit', 'Used, acceptable: Last visit', 'Used, acceptable: Lowest', 'Used, acceptable: Highest', 'Warehouse Deals: Current', 'Warehouse Deals: 30 days avg.', 'Warehouse Deals: 90 days avg.', 'Warehouse Deals: 180 days avg.', 'Warehouse Deals: 1 day drop %', 'Warehouse Deals: 7 days drop %', 'Warehouse Deals: 30 days drop %', 'Warehouse Deals: 90 days drop %', 'Warehouse Deals: Drop since last visit', 'Warehouse Deals: Drop % since last visit', 'Warehouse Deals: Drop since last visit.1', 'Warehouse Deals: Lowest', 'Warehouse Deals: Highest', 'Refurbished: Current', 'Refurbished: 30 days avg.', 'Refurbished: 90 days avg.', 'Refurbished: 180 days avg.', 'Refurbished: 1 days drop %', 'Refurbished: 7 days drop %', 'Refurbished: 30 days drop %', 'Refurbished: 90 days drop %', 'Refurbished: Drop since last visit', 'Refurbished: Drop % since last visit', 'Refurbished: Last visit', 'Refurbished: Lowest', 'Refurbished: Highest', 'Collectible: Current', 'Collectible: 30 days avg.', 'Collectible: 90 days avg.', 'Collectible: 180 days avg.', 'Collectible: 1 day drop %', 'Collectible: 7 days drop %', 'Collectible: 30 days drop %', 'Collectible: 90 days drop %', 'Collectible: Drop since last visit', 'Collectible: Drop % since last visit', 'Collectible: Last visit', 'Collectible: Lowest', 'Collectible: Highest', 'List Price: Current', 'List Price: 30 days avg.', 'List Price: 90 days avg.', 'List Price: 180 days avg.', 'List Price: 1 day drop %', 'List Price: 7 days drop %', 'List Price: 30 days drop %', 'List Price: 90 days drop %', 'List Price: Drop since last visit', 'List Price: Drop % since last visit', 'List Price: last visit', 'List Price: Lowest', 'List Price: Highest', 'Trade-In: Current', 'Trade-In: 30 days avg.', 'Trade-In: 90 days avg.', 'Trade-In: 180 days avg.', 'Trade-In: 1 day drop %', 'Trade-In: 7 days drop %', 'Trade-In: 30 days drop %', 'Trade-In: 90 days drop %', 'Trade-In: Drop since last visit', 'Trade-In: Drop % since last visit', 'Trade-In: Last visit', 'Trade-In: Lowest', 'Trade-In: Highest', 'Rental: Current', 'Rental: 30 days avg.', 'Rental: 90 days avg.', 'Rental: 180 days avg.', 'Rental: 1 day drop %', 'Rental: 7 days drop %', 'Rental: 30 days drop %', 'Rental: 90 days drop %', 'Rental: Drop since last visit', 'Rental: Drop % since last visit', 'Rental: Last visit', 'Rental: Lowest', 'Rental: Highest', 'eBay New: Current', 'eBay New: 30 days avg.', 'eBay New: 90 days avg.', 'eBay New: 180 days avg.', 'eBay New: 1 day drop %', 'eBay New: 7 days drop %', 'eBay New: 30 days drop %', 'eBay New: 90 days drop %', 'eBay New: Drop since last visit', 'eBay New: Drop % since last visit', 'eBay New: Last visit', 'eBay New: Lowest', 'eBay New: Highest', 'eBay Used: Current', 'eBay Used: 30 days avg.', 'eBay Used: 90 days avg.', 'eBay Used: 180 days avg.', 'eBay Used: 1 day drop %', 'eBay Used: 7 days drop %', 'eBay Used: 30 days drop %', 'eBay Used: 90 days drop %', 'eBay Used: Drop since last visit', 'eBay Used: Drop % since last visit', 'eBay Used: Last visit', 'eBay Used: Lowest', 'eBay Used: Highest', 'New Offer Count: Current', 'New Offer Count: 30 days avg.', 'New Offer Count: 90 days avg.', 'New Offer Count: 180 days avg.', 'New Offer Count: 1 day drop %', 'New Offer Count: 7 days drop %', 'New Offer Count: 30 days drop %', 'New Offer Count: 90 days drop %', 'Count of retrieved live offers: New, FBA', 'Count of retrieved live offers: New, FBM', 'Used Offer Count: Current', 'Used Offer Count: 30 days avg.', 'Used Offer Count: 90 days avg.', 'Used Offer Count: 180 days avg.', 'Used Offer Count: 1 day drop %', 'Used Offer Count: 7 days drop %', 'Used Offer Count: 30 days drop %', 'Used Offer Count: 90 days drop %', 'Refurbished Offer Count: Current', 'Refurbished Offer Count: 30 days avg.', 'Refurbished Offer Count: 90 days avg.', 'Refurbished Offer Count: 180 days avg.', 'Refurbished Offer Count: 1 day drop %', 'Refurbished Offer Count: 7 days drop %', 'Refurbished Offer Count: 30 days drop %', 'Refurbished Offer Count: 90 days drop %', 'Collectible Offer Count: Current', 'Collectible Offer Count: 30 days avg.', 'Collectible Offer Count: 90 days avg.', 'Collectible Offer Count: 180 days avg.', 'Collectible Offer Count: 1 day drop %', 'Collectible Offer Count: 7 days drop %', 'Collectible Offer Count: 30 days drop %', 'Collectible Offer Count: 90 days drop %', 'Tracking since', 'Listed since', 'URL: Amazon', 'URL: Keepa', 'Categories: Root', 'Categories: Sub', 'Categories: Tree', 'Launchpad', 'ASIN', 'Product Codes: EAN', 'Product Codes: PartNumber', 'Parent ASIN', 'Variation ASINs', 'Freq. Bought Together', 'Manufacturer', 'Brand', 'Product Group', 'Model', 'Variation Attributes', 'Color', 'Size', 'Edition', 'Format', 'Author', 'Contributors', 'Binding', 'Number of Items', 'Number of Pages', 'Publication Date', 'Release Date', 'Languages', 'Package: Dimension (cm³)', 'Package: Length (cm)', 'Package: Width (cm)', 'Package: Height (cm)', 'Package: Weight (g)', 'Package: Quantity', 'Item: Dimension (cm³)', 'Item: Length (cm)', 'Item: Width (cm)', 'Item: Height (cm)', 'Item: Weight (g)', 'Adult Product', 'Trade-In Eligible', 'Prime Eligible (Amazon offer)', 'Subscribe and Save', 'One Time Coupon: Absolute', 'One Time Coupon: Percentage', 'Subscribe and Save Coupon: Percentage', 'Unnamed: 416', 'Unnamed: 417', 'Unnamed: 418', 'link', 'csv', 'stats', 'categories', 'imagesCSV', 'manufacturer', 'title', 'lastUpdate', 'lastPriceChange', 'rootCategory', 'productType', 'parentAsin', 'variationCSV', 'asin', 'domainId', 'type', 'brand', 'productGroup', 'partNumber', 'model', 'color', 'size', 'format', 'packageHeight', 'packageLength', 'packageWidth', 'packageWeight', 'packageQuantity', 'binding', 'numberOfItems', 'eanList', 'upcList', 'frequentlyBoughtTogether', 'features', 'description', 'promotions', 'coupon', 'availabilityAmazon', 'fbaFees', 'variations', 'itemHeight', 'itemLength', 'itemWidth', 'itemWeight', 'g', 'categoryTree', 'stats_parsed', 'stats_parsed_1', 'stats_parsed_2', 'packageWeight2', 'minn', 'mid', 'maxx', 'bsr_c', 'bsr']







df = pd.DataFrame(
    [
        {"command": "st.selectbox", "rating": 4, "is_widget": True},
        {"command": "st.balloons", "rating": 5, "is_widget": False},
        {"command": "st.time_input", "rating": 3, "is_widget": True},
    ]
)


a,b = st.tabs(["a","b"])
with a:
    edited_df = st.data_editor(
    df,
    column_config={
        "command": "Streamlit Command",
        "rating": st.column_config.NumberColumn(
            "Your rating",
            help="How much do you like this command (1-5)?",
            min_value=1,
            max_value=5,
            step=1,
            format="%d ⭐",
        ),
        "is_widget": "Widget ?",
    },
    disabled=[item for item in df.columns if item != "is_widget"],

    hide_index=True,
)

# favorite_command = edited_df.loc[edited_df["rating"].idxmax()]["command"]
# st.markdown(f"Your favorite command is **{favorite_command}** 🎈")

with b:
    
    edited_df[edited_df['is_widget']==True]
