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

product_keys = ["categories", "imagesCSV","manufacturer","title","lastUpdate","lastPriceChange","rootCategory", "productType","parentAsin","variationCSV","asin","domainId","type","brand","productGroup","partNumber","model","color","size","format","packageHeight","packageLength","packageWidth","packageWeight","packageQuantity","binding","numberOfItems","eanList","upcList","frequentlyBoughtTogether","features","description","promotions","coupon","availabilityAmazon","fbaFees","variations","itemHeight","itemLength","itemWidth","itemWeight","g","categoryTree","stats_parsed"]

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
@st.cache_data 
def bb_price(param):
    if pd.isna(param):
       return "NO BB"
    elif  "buyBoxPrice" in param.keys():
        return round(param["buyBoxPrice"]/ 100, 2)
    else: 
        return None

            # ---------------------------------------------------------------------------------------

# buybox price 2
@st.cache_data 
def bb_price_2(param):
    if pd.isna(param):
       return "NO BB"
    elif  "buyBoxPrice" in eval(param).keys():
        return  round((int(eval(param)["buyBoxPrice"]) / 100), 2)
    else: 
        return None
    
@st.cache_data 
def get_bb_price(bb_price, df_Wsku):
    df_Wsku["bb_price"] = df_Wsku["stats_parsed"].apply(bb_price)   

# ---------------------------------------------------------------------------------------

# bb stats info
@st.cache_data 
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
@st.cache_data 
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


@st.cache_data 
def get_bb_stats(bb_stats, df_Wsku, day, stats):
    features = ["stats" + str(day) + "_bb_stats","stats" + str(day) + "_bb_seller_number"]
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
@st.cache_data 
def get_dim(df_Wsku):
    df_Wsku["packageHeight"]= df_Wsku["packageHeight"].apply(lambda x: round(x *  0.0393701, 2))
    df_Wsku["packageLength"]= df_Wsku["packageLength"].apply(lambda x: round(x * 0.0393701, 2))
    df_Wsku["packageWidth"]= df_Wsku["packageWidth"].apply(lambda x: round(x * 0.0393701,2))
    df_Wsku["packageWeight2"]= df_Wsku["packageWeight"].apply(lambda x: round(x  *  0.035274,2)) 
    df_Wsku["packageWeight"]= df_Wsku["packageWeight"].apply(lambda x: round(x *  0.00220462,2))
    df_Wsku["minn"] = [j.sort_values()[0] if pd.isna(j).any() == False else None for i,j in df_Wsku[["packageHeight","packageWidth", "packageLength"]].iterrows()]
    df_Wsku["mid"] = [j.sort_values()[1] if pd.isna(j).any() == False else None for i,j in df_Wsku[["packageHeight","packageWidth", "packageLength"]].iterrows()]
    df_Wsku["maxx"] = [j.sort_values()[2] if pd.isna(j).any() == False else None  for i,j in df_Wsku[["packageHeight","packageWidth", "packageLength"]].iterrows()]

# ---------------------------------------------------------------------------------------
@st.cache_data 
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
    
@st.cache_data 
def get_roi(df_Wsku, prod_type):
    features = ["roi", "margin", "profit", "total_fee", "FBA_fee", "referral_fee", "unit_F_cost", "storage_std", "storage_ovs","low_price"]
    for i in range(0,len(features)):
        df_Wsku[features[i]] = [low_price(k,prod_type)[i] for p,k in df_Wsku[['bb_price','minn', 'mid', 'maxx', 'packageHeight', 'packageLength', 'packageWidth', 'packageWeight', 'packageWeight2',"Coral Price"]].iterrows()]

# ---------------------------------------------------------------------------------------

# oos info
@st.cache_data 
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
@st.cache_data  
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

@st.cache_data 
def get_oos(oos, df_Wsku, day, stats):
    features = ["stats" + str(day) + "_oos30","stats" + str(day) + "_oos90"]
    print(features)
    for i in range(len(features)):
        df_Wsku[features[i]] = df_Wsku[stats].apply(oos).apply(lambda x: x[i])

# ---------------------------------------------------------------------------------------

# bsr info
@st.cache_data 
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
@st.cache_data 
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
@st.cache_data 
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

st.title('SmartScanner')



    

































































































uploaded_files = st.file_uploader("Choose files (csv or xlsx)", accept_multiple_files=True)

if uploaded_files:
    df_uploaded = process_uploaded_file(uploaded_files[0])
    df_coral = process_uploaded_file(uploaded_files[1])
    df_upc = process_uploaded_file(uploaded_files[2])
    # Now you can use df_uploaded, df_coral, and df_upc with caching applied.
    df_uploaded = df_uploaded.dropna(subset='Product Codes: UPC').reset_index(drop=True)
    df_uploaded["link"] = "https://www.amazon.com/dp/" + df_uploaded["ASIN"] 

    df_coral = df_coral.drop_duplicates(subset="asin")
    df_coral.columns = [i.upper() for i in list(df_coral.columns)]

    df_uploaded = df_uploaded.merge(df_coral[["ASIN","CORALPORTID"]], how = "left", on = "ASIN").loc[:,["CORALPORTID"]+ list(df_uploaded.columns)].sort_values(by = "Sales Rank: Current").drop_duplicates(subset = "ASIN")
    st.write("df.shape : ", df_uploaded.shape)
    st.write("Sitede yuklu urun sayisi : ", df_uploaded["CORALPORTID"][~df_uploaded["CORALPORTID"].isna()].count())
    

    df_200k = df_uploaded[df_uploaded["Sales Rank: Current"] < 150000].reset_index(drop = True)
    st.write("df_200k.shape : ",df_200k.shape)
    st.write("Sitede yuklu urun sayisi : ", df_200k["CORALPORTID"][~df_200k["CORALPORTID"].isna()].count())
    st.write("Total 200k alti urun sayisi : ",df_200k.shape[0])


    df_Wsku = match(df_200k, df_upc)

    df_Wsku = df_Wsku.merge(df_200k, how = "left", on = "Product Codes: UPC").drop_duplicates(subset="ASIN").reset_index(drop = True)
    st.write("df_Wsku.shape : ",df_Wsku.shape)
    st.write("200k alti bizim satabilecegimiz urun sayisi : ",df_Wsku.shape[0])

    num_rows = len(df_Wsku)
    if num_rows > 100:
        st.header(f"{num_rows} adet ürün bulundu. Malesef max 300 adet ürün bilgilerini alabilirsiniz!!")

    # Kullanıcıya hangi parti seçmesi gerektiğini sorun
    selected_chunk = st.selectbox("Hangi parta ait verileri görmek istersiniz?", range(1, (num_rows // 100) + 2))

    # Seçilen partiye ait verileri görüntüleyin
    start_row = (selected_chunk - 1) * 100
    end_row = min(selected_chunk * 100, num_rows)
    st.write(f"Selected part of products:  {selected_chunk}    >>>    {start_row}-{end_row}")
    df_Wsku[start_row: end_row]
    output_0 = BytesIO()
    df_Wsku.to_excel(output_0, index=False, engine='openpyxl')
    output_0.seek(0)

    # İndirme bağlantısını göster
    b64 = base64.b64encode(output_0.read()).decode()
    st.markdown(f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="main_file.xlsx">Download main file!</a>', unsafe_allow_html=True)
    
    
    st.success('Analysis is complete!', icon="✅")

    api = keepa.Keepa(accesskey)
    api.update_status()
    current_token = api.tokens_left
    st.write("current_token",current_token)



    # Kullanıcıdan seçenekleri seçmesini isteyin
    selected_option = st.selectbox("Lütfen bir seçenek seçin:", [30, 90, 180, 360])
    state_dict = dict()

    for s_option in  [30, 90, 180, 360]:
        for s_chunk in list(range(1,(num_rows // 100) + 2)):
            state_dict[f"{s_chunk}_{s_option}"] = pd.DataFrame(df_Wsku[start_row: end_row].index)
   

    if 'state_dict' not in st.session_state:
        st.session_state.state_dict = state_dict

    analysis_button = st.button("Start Analysis")
    analysis_message = st.empty()

    if analysis_button:
        
        api = keepa.Keepa(accesskey)
        api.update_status()
        current_token = api.tokens_left
        st.write("current_token",current_token) 

        if current_token > 9000:
            analysis_message.text("Analysis in progress...")
            result = send_req_threading(df_Wsku[start_row: end_row],selected_option)
        
            df_result = pd.DataFrame(result, columns = product_keys)
            df_result
            
            st.session_state.state_dict[f"{selected_chunk}_{selected_option}"] = df_result
            
            api = keepa.Keepa(accesskey)
            api.update_status()
            current_token = api.tokens_left
            st.write("current_token",current_token)


            xlsx = df_result.to_csv(index=False)
            # CSV verisini base64'e dönüştürün
            b64 = base64.b64encode(xlsx.encode()).decode()

        # İndirme bağlantısını oluşturun
            href = f'<a href="data:file/xlsx;base64,{b64}" download="{selected_chunk}_resutl_{selected_option}.csv">CSV Dosyasını İndir!</a>'
            st.markdown(href, unsafe_allow_html=True)

        # DataFrame'i bir XLSX dosyasına kaydet
            output = BytesIO()
            df_result.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)

            # İndirme bağlantısını göster
            b64 = base64.b64encode(output.read()).decode()
            st.markdown(f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{selected_chunk}_result_{selected_option}.xlsx">xlsx Dosyasını İndir!</a>', unsafe_allow_html=True)
            analysis_message.text("Analysis is complete!")
            
            st.success('Analysis is complete!', icon="✅")


        else:
            st.write("Current token yeterli degil. Lütfen 30dk sonra tekrar deneyin!!!")

    if st.button("Current Token"):
        api = keepa.Keepa(accesskey)
        api.update_status()
        current_token = api.tokens_left
        st.write("current_token",current_token)

    repr_data = dict()
    
    for h,g in st.session_state.state_dict.items():
        if "stats_parsed" in st.session_state.state_dict[h].keys():
            repr_data[h] = True
        else:
            repr_data[h] = False

    
    st.write(pd.DataFrame({"data": repr_data}).T)

    #if st.success("Add result main list!"):
    a1 = pd.concat([st.session_state.state_dict["1_30"], st.session_state.state_dict["2_30"]]).reset_index(drop=True)
    b1 = pd.concat([st.session_state.state_dict["1_90"], st.session_state.state_dict["2_90"]]).reset_index(drop=True)
    c1= pd.concat([st.session_state.state_dict["1_180"], st.session_state.state_dict["2_180"]]).reset_index(drop=True)
    d1 = pd.concat([st.session_state.state_dict["1_360"], st.session_state.state_dict["2_360"]]).reset_index(drop=True)
   
    
    b1.columns = [str(i) + "_90" for i in b1.columns]
    c1.columns = [str(i) + "_180" for i in c1.columns]
    d1.columns = [str(i) + "_360" for i in d1.columns]
    st.write("b1",b1)
    #d1.columns = [i + "_360" for i in d1.columns]
 

    if 'result_data' not in st.session_state:
        st.session_state.result_data = pd.concat([a1,b1,c1,d1], axis = 1)



    st.session_state.result_data = pd.concat([a1,b1,c1,d1], axis = 1)
    st.write("st.session_state.result_data", st.session_state.result_data)
  

    
    
    # a1 = pd.concat([st.session_state.state_dict["1_30"], st.session_state.state_dict["2_30"],st.session_state.state_dict["3_30"], st.session_state.state_dict["4_30"]]).reset_index(drop=True)
    # b1 = pd.concat([st.session_state.state_dict["1_90"], st.session_state.state_dict["2_90"],st.session_state.state_dict["3_90"], st.session_state.state_dict["4_90"]]).reset_index(drop=True)
    # c1= pd.concat([st.session_state.state_dict["1_180"], st.session_state.state_dict["2_180"],st.session_state.state_dict["3_180"], st.session_state.state_dict["3_180"]]).reset_index(drop=True)
    # #d1 = pd.concat([st.session_state.state_dict["1_360"], st.session_state.state_dict["2_360"],st.session_state.state_dict["3_360"], st.session_state.state_dict["4_360"]]).reset_index(drop=True)
    # st.write(c1)
    
    # try:
        
       
    #     a1.columns = [i + "_30" for i in a1.columns]
    #     b1.columns = [i + "_90" for i in b1.columns]
    #     c1.columns = [i + "_180" for i in c1.columns]
    #     #d1.columns = [i + "_360" for i in d1.columns]
    #     st.write(pd.concat([a1,b1,c1]))

    #     st.write(pd.concat([a1,b1,c1]))
    # except:
    #     st.warning("Please add a require data!!!")
    #     st.write("Please add a require data!!!")



# state_df_dict = dict()

# for key in all_keys:
#     duration_df = pd.DataFrame()
#     parts = key.split('_')
#     value = int(parts[0])
#     duration = int(parts[1])
#     duration_df = ([duration_df, state_dict[f"{value}_{duration}"]])
#     duration_df.columns = [i + f"_{duration}" for i in duration_df.columns]
#     state_df_dict[duration] = pd.DataFrame()
#     state_df_dict[duration] = pd.concat([state_df_dict[duration], state_dict[f"{value}_{duration}"]])
#     st.write(f"{value}_{duration}")
# st.write(state_df_dict)

# st.write(pd.concat(df_Wsku, state_df_dict))


# g = {30:pd.DataFrame(),90:pd.DataFrame(),180:pd.DataFrame(),360:pd.DataFrame()}
# for i in all_keys:
#     for c,o in [i.split("_")]:
        
#         if o == 30:

#             g[30] = pd.concat([g[30],state_dict[f"{c}_{o}"]]) 
#             st.write(f"{c}_{o}")
#             state_dict[f"{c}_{o}"]
# st.write(g)



