import pandas as pd
import nltk
from nltk.corpus import stopwords
import math
import math
import json
import requests
from nltk.tokenize import word_tokenize
import re
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from scipy import spatial
import time

BERT_BASE_MODEL = SentenceTransformer("bert-base-nli-mean-tokens")


class AddressMatching:
    def __init__(self, df, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.stop_words = set(stopwords.words("english"))
        self.df = df
        self.addresses = df["address"]
        self.corpus_ = []
        self.tf_ = []
        self.df_ = {}
        self.idf_ = {}
        self.doc_len_ = []
        self.corpus_size_ = 0
        self.avg_doc_len_ = 0
        self.abbrevation_dict = {
            "rd": "road",
            "st": "street",
            "ave": "avenue",
            "hwy": "highway",
            "bldg": "building",
            "ctr": "centre",
            "cir": "circles",
            "flt": "flat",
            "hts": "heights",
            "ln": "lane",
            "pt": "point",
            "pvt.": "private",
            "co": "corporation",
            "corp": "corporation",
            "ltd": "limited",
            "blvd": "boulevard",
            "crt": "court",
            "cres": "crescent",
            "dr": "drive",
            "pl": "place",
            "sq": "square",
            "stn": "station",
            "terr": "terrace",
            "pkwy": "packingway",
            "usa": "united states of america",
            "us": "united states",
        }

    def __pre_process_address(self):
        self.corpus_ = [
            [
                word
                for word in self.__handle_abbreviation(document.replace("\n", " "))
                .replace(",", " ")
                .split()
                if word not in self.stop_words
            ]
            for document in self.addresses
        ]

    def __handle_abbreviation(self, address):
        north_chk = [
            (i.group(), i.start(), i.end())
            for i in re.finditer(r"[,\s]?\d{3,5}\sN\s", address)
        ]
        if north_chk:
            grp, start, end = north_chk[0]
            address = address[:start] + " " + grp.split()[0] + " NORTH " + address[end:]
        south_chk = [
            (i.group(), i.start(), i.end())
            for i in re.finditer(r"[,\s]?\d{3,5}\sS\s", address)
        ]
        if south_chk:
            grp, start, end = south_chk[0]
            address = address[:start] + " " + grp.split()[0] + " SOUTH " + address[end:]
        east_chk = [
            (i.group(), i.start(), i.end())
            for i in re.finditer(r"[,\s]?\d{3,5}\sE\s", address)
        ]
        if east_chk:
            grp, start, end = east_chk[0]
            address = address[:start] + " " + grp.split()[0] + " EAST " + address[end:]
        west_chk = [
            (i.group(), i.start(), i.end())
            for i in re.finditer(r"[,\s]?\d{3,5}\sW\s", address)
        ]
        if west_chk:
            grp, start, end = west_chk[0]
            address = address[:start] + " " + grp.split()[0] + " WEST " + address[end:]

        address_words = address.replace(".", "").replace(",", " ").lower().split()
        for word_id, word in enumerate(address_words):
            if word in self.abbrevation_dict:
                address_words[word_id] = self.abbrevation_dict[word]

        return " ".join(address_words)

    def __score(self, query, index):
        score = 0.0

        doc_len = self.doc_len_[index]
        frequencies = self.tf_[index]
        for term in query:
            if term not in frequencies:
                continue

            freq = frequencies[term]
            numerator = self.idf_[term] * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (
                1 - self.b + self.b * doc_len / self.avg_doc_len_
            )
            score += numerator / denominator

        return score

    def __check_string_similarity(self, line_items: list, score: float = 0.55) -> bool:
        if len(line_items) != 2:
            return False
        sentence_embeddings = BERT_BASE_MODEL.encode(line_items)
        # print(1 - spatial.distance.cosine(sentence_embeddings[0], sentence_embeddings[1]))
        return (
            1
            - spatial.distance.cosine(sentence_embeddings[0], sentence_embeddings[1])
            # >= score
        )

    def fit(self):
        self.__pre_process_address()

        for document in self.corpus_:
            self.corpus_size_ += 1
            self.doc_len_.append(len(document))

            # compute tf (term frequency) per document
            frequencies = {}
            for term in document:
                term_count = frequencies.get(term, 0) + 1
                frequencies[term] = term_count

            self.tf_.append(frequencies)

            # compute df (document frequency) per term
            for term, _ in frequencies.items():
                df_count = self.df_.get(term, 0) + 1
                self.df_[term] = df_count

        for term, freq in self.df_.items():
            self.idf_[term] = math.log(
                1 + (self.corpus_size_ - freq + 0.5) / (freq + 0.5)
            )

        self.avg_doc_len_ = sum(self.doc_len_) / self.corpus_size_
        return self

    def search(self, query):
        query = query.lower()
        query_list = [word for word in query.split() if word not in self.stop_words]
        scores = [self.__score(query_list, index) for index in range(self.corpus_size_)]

        scores, result = zip(*sorted(zip(scores, self.addresses), reverse=True))

        threshold = 1

        wgt_dict = {
            "name": 0.21,
            "streetName": 0.23,
            "city": 0.19,
            "state": 0.16,
            "zipCode": 0.11,
            "extnZip": 0.0,
            "country": 0.1,
        }
        output = ""
        category = ""
        if scores[0] - scores[1] >= threshold:
            output = "\n" + result[0]
            score = self.__check_string_similarity([result[0], query])
            if score > 0.90:
                category = "Very High"
            elif score > 0.80:
                category = "High"
            elif score < 0.70:
                category = "Low"
            else:
                category = "Medium"
        else:
            final_result = ""
            highest_score = 0
            for add in result[0:3]:
                score = self.__check_string_similarity([add, query])
                # add_df = self.df[self.df["address"] == add]
                # score = 0
                # for element in wgt_dict:
                #     content = add_df[element].to_string().lower()
                #     if element in ["streetName","name"]:
                #         # print(element)
                #         if (
                #             self.__check_string_similarity([content,query])
                #         ):
                #             score += wgt_dict[element]
                #     else:
                #         if content in query :
                #             score += wgt_dict[element]

                if score > highest_score:
                    highest_score = score
                    final_result = add
            output = "\n" + final_result
            if highest_score > 0.90:
                category = "High"
            elif highest_score > 0.85:
                category = "Medium"
            else:
                category = "Low"

        return category, output


def convert_to_list(filepath):
    with open(filepath) as file:
        data = json.load(file)
    df = pd.DataFrame.from_dict(data)

    addresses = []
    for name, street_name, city, state, zipCode, extnZip, country in zip(
        df["name"],
        df["streetName"],
        df["city"],
        df["state"],
        df["zipCode"],
        df["extnZip"],
        df["country"],
    ):
        add = [name, street_name, city, state, zipCode + "" + extnZip, country]
        add = [element for element in add if element != ""]
        addresses.append(", ".join(add))

    df["address"] = addresses
    return df


originAddress = convert_to_list("calhounorders-originAddress.json")
returnAddress = convert_to_list("calhounorders-returnAddress.json")
deliveryAddress = convert_to_list("calhounorders-deliveryAddress.json")


bm25_originAddress = AddressMatching(originAddress)
bm25_returnAddress = AddressMatching(returnAddress)
bm25_deliveryAddress = AddressMatching(deliveryAddress)

bm25_originAddress.fit()
bm25_returnAddress.fit()
bm25_deliveryAddress.fit()


delivery_queries = [
    "MIDWEST MANUFACTURING/MW PREHUNG PLANT – BLDG #320,14320 COUNTY ROAD 15, HOLIDAY CITY, OH 43554,TEL: 419-485-6584,ATTN: JUSTIN CRAWFORD eMAIL: jcrawfor@midwestmanufacturing.com",
    "Alliance Outdoor Group Inc.4949 264th StValley, NE, 68064, United States",
    "Admiral Moving & Logistics, C/O Corrigan Logistics Graduate Fayetteville, 1245 E Henri De Tonti Blvd Springdale, AR 72762 -Ctc: Colton Gregory",
    "QUIN GLOBAL US, INC.  5710 F STREET OMAHA, NE 68117 TEL :402-731-3636 Hours: M-F 8:00 AM to 3:00 PM",
    "Chainworks, Inc. 3255 Hart Rd Jackson, MI 49201",
    "MID-STATE WAREHOUSE 888 O Neill Drive Hebron Oh, 43025 Justin Carder PH: 740-929-5130",
    "NIPPON EXPRESS HEMLOCK WH 10401 HARRISON RD STE 101 ROMULUS, MI 48174 TIM ACKER 734-740-8534",
    "HANON C/O AEL SPAN 41775-100 ECORSE RD BELLEVILLE MI 48111 UNITED STATES",
    "HOG SLAT INC 1112 20TH STREET NORTH HUMBOLDT, IA 50548, US",
    "Pivot International, Inc. C/O Election Systems & Software, In 11208 John Galt Blvd Omaha NE 68137",
    "SIEMENS GAMESA RENEWABLE ENERGY, INC. 714 COREY RD. HUTCHINSON KS 67501 USA ATTN: RANDY AVERY PH#: 620 259 7424 MOB: 620 314 5909 RANDY.AVERY@SIEMENS.COM",
    "MENARDS # 3439 7422 EAST STREET Shelby, IA 51570",
    "GATES CORPORATION 3015 LEMONE INDUSTRIAL BLVD COLUMBIA MO 65201",
    "The Coleman Company 2111 E 37th St N, Wichita, KS 67219",
    "W3167 County Rd S Iron Ridge WI 53035 United States",
    "1701 PIERCE BUTLER RTE SAINT PAUL MN 55104 United States",
    "MYGRANT GLASS COMPANY INC. 44 NORTHERN STACKS DR SUITE 100 FRIDLEY MN 55421 USA",
    "MURPHY WAREHOUSE, 701 24TH AVE SE, DOCK DOOR #33 Minneapolis, MN 55413",
    "MURPHY WAREHOUSE,701 24TH AVE SE, DOCK DOOR #33 Minneapolis, MN 55413",
    "ASP TRADING, LLC. Kollektornaya Street 30, Office 2, Kyiv, 02660 Ukraine",
    "DS INC. DC# 3439 7422 EAST STREET SHELBY, IA 51570 PO# 0506792-78673888, 0506791-78673877,0575745-78827602",
    "NORGOLD - NEW ADDRESS 3770 Hagen Drive SE Wyoming, MI49548, UNITED STATES TEL: 616-988-0880",
    "HIGHLIGHT INDUSTRIES INC 2694 PRAIRIE STREET SW WYOMING, MI, 49519, US eMail: RENEEV@HIGHLIGHTINDUSTRIES.COM Tel: 616-531-2464 Fax: 616-531-0506 Contact: RENEE VAN GILST",
    "BNSF Logistics Park, 26664 Baseline Road, Elwood IL 60421,  U.S.A. , Customer Service 630-595-3770 ",
    "F & P AMERICA 2101 CORPORATION DRIVE, TROY, OH 45373 Contact: Jason Ratliff 937-570-9132",
    "TROPICAL NUT & FRUIT  3150 URBANCREST INDUSTRIAL DRIVE UNBANCREST, OH 43123 905-812-8960, DEBBIE CARINS  TROPICALFREIGHT@ROGERS.COM PO: P 29658 RED RIVER FOODS",
    "SCHWARZ BP121 6000 GREEN POINTE DR S GROVEPORT OH 43125 SELECT JONA SCOTT: 6144918560 X 1300 614-491-8560 W/H JONA.SCOTT@SCHWARZ.COM",
    "VAL PRODUCTS 200 W SYCAMORE 45828 COLDWATER, OH USA LILIANA TEL. 419-678-8731 X 261 LINDA TEL. 419-678-8731 X 208 REFERENCE: PO# 119236-00",
    "MENARD, INC. 14311 COUNTY ROAD 15 HOLIDAY CITY OH 43554",
    "K & K Die Inc. 40700 Enterprise Dr. Dmartinez@kandkdie.com STERLING HEIGHTS, MI 48314",
    "PLUNKETT DIST - FORT SMITH 1010 SOUTH Y STREET, FORT SMITH, AR 72901",
    "Torque Fitness 9365 Holly St NW COON RAPIDS, MN 55433-5807 United States",
    "CHEROKEE CARPET INDUSTRIES 601 CALLAHAN ROAD PO BOX 3577 706-277-6277",
    "EMPIRE HOME FASHION 3080 WOODMAN DR KETTERING COLUMBUS, OH, 45420, US eMail: EMPIREINVOICE30@GMAIL.COM Tel: 973-873-4645 Contact: ABDEL HAMID AWAD",
    "SIEMENS GAMESA RENEWABLE ENERGY, INC. 714 COREY RD. HUTCHINSON KS 67501 USA ATTN: RANDY AVERY PH#: 620 259 7424 MOB: 620 314 5909 RANDY.AVERY@SIEMENS.COM",
    "KANSAS EROSION,LLC 3600 AIRPORT RD.,SALINA,KS 67401 Contact:Cathy Zecha Phone:620-617-9220 Email:Kansaserosion2@gmail.com",
    "CLEARVIEW ENTERPRISES 451 AGNES DR. Tontitown, AR 72770 United States",
    "CENTRAL LOGISTIC SERVICES, INC 1850 CENTENNIAL AVENUE HASTINGS, NE 68901 402-461-3775",
    "Menards #9008 7421 EAST STREET Shelby, IA 51570 United States",
    "GATES CORPORATION 3015 LEMONE INDUSTRIAL BLVD COLUMBIA MO 65201",
    "FLATBED SERVICES 2260 ANDREW AVE SERGEANT BLUFF IA 51054 TEL :712-943-2030",
    "GOLD STONE TRADING INC 29330 STEPHENSON HWY MADISON HEIGHTS, MI 48071 T: 248-929-1868 CTC: RICKY",
    "MENARDS 14502 COUNTY ROAD 15 Holiday City, OH 43554",
    "MENARDS #3339 14502 COUNTY RD 15 Holiday City, OH 43554",
    "FENCHEM INC 1400 SE GATEAWAY DR. #101 GRIMES, IA 50111 ATTN.:AIHUA FEI (EXT#501)",
    "CENTRAL LOGISTIC SERVICES, INC 1850 CENTENNIAL AVENUE HASTINGS, NE 68901 402-461-3775",
    "MENARD INC, 14502 COUNTY ROAD, HOLIDAY CITY OH 43554-8705, U.S.A",
    "MENARDS INC SHELL ROCK CROSSDOCK # 9028 22281 WRANGLER RD SHELL ROCK, IA 50670",
    "MENARDS - EAU CLAIRE, 5103 NORTH TOWN HALL ROAD EAU CLAIRE WI 54703, REF MENARDS PO# IN NOTES 715-876-2600",
    "GLOBAL LINK SOURCING C/O GENERAL PAPER PRODUCTS 6650 143RD AVENUE NW RAMSEY MN 55303 SHANNON MINSTER 763 323-8389",
    "INTERCEPT 34621 STATE HIGHWAY 11 ROSEAU MN 56751 UNITED STATES",
    "MENARD INC MCKENZIE CROSSDOCK #9022 24461 CPUNTY HWY 10 BUILDING 921 MCKENZIE, ND 58572",
    "BENCHMARK ELECTRONICS INC 4155 THEURER BLVD WINONA, MN 55987 UNITED STATES OF AMERICA TEL: 507-453-4877",
    "ATLAPAC COMMERCIAL BAG COMPANY 2901 E 4TH AVE COLUMBUS OH 43219 UNITED STATES",
    "MENARDS 7422 EAST STREET Shelby, IA 51570",
    "NORFOLK SOUTHERN RAMP, 4800 N KIMBALL DR, KANSAS CITY MO 64161, UNITED STATES",
    "PONY EXPRESS WAREHOUSE // ARYSTA LIFESCIENCE, 2307 ALABAMA ST, SAINT JOSEPH MO 64504, UNITED STATES",
    "ASHLEY FURNITURE INDUSTRIES ONE ASHLEY WAY ARCADIA, WI 54612 Tel : 920 757 9590 Fax : Contact : DAN KRAY",
    "Preferred Chicago 32357 S. Wood St Chicago, IL 60608 T: (773) 268-3400",
    "Preferred Chicago 32357 S. Wood St.Chicago, IL 60608 T: (773) 268-3400",
    "ALDI INC., 1200 N KIRK RD BATAVIA, IL 60510-1477,USA",
    "Sandy Francis Build SMART 3701 Greenway Circle Lawrence, KS 66046 Email: sfrancis@buildsmartna.com 785-331-1022",
    "FEDEX TRADE NETWORKS-MI 11101 METRO AIRPORT CENTER DR., STE. 100 ROMULUS, MI 48174-1694, UNITED STATES MAHER HARHARA TEL: 313.518.3100 MIKE 810-650-6136/966-3603",
    "WOODCO, 3140 103RD LN NE BLAINE MN 55449-4522 UNITED STATES",
    "MOLDED FIBERGLASS COMPANIES 1401 BROWN COUNTY RD. 19 NORTH ABERDEEN SD 57401 USA",
    "MENARD INC, HOLIDAY CITY DC, 14502 COUNTY ROAD 15, HOLIDAY CITY OH 43554, U.S.A",
    "SOFIDEL - TULSA 16400 EAST 620 ROAD INOLA, OK 74036",
    "Protect Land and Equipment LLC c/o Rheal Cattle, 209 E Jackson St., Fremont NE 68025 Tel : 402-721-9525",
    "MENARDS MENARDS - YARD # 3039 4860 MENARD DR MENARDS DC SOUTH EAU CLAIRE, WI 54703 UNITED STATES OF AMERICA",
    "SHELBY 3439 7422 EAST STREET SHELBY, IA 51570-3320",
    "SWAN CREEK CANOE CO. 395 WEST AIRPORT HIGHWAY SWANTON, OH 43558 CTC: KATHERINE DUET / T: 419-825-1151",
    "MENARD INC. 4860 MENARD DRIVE EAU CLAIRE, WI 54703-9604, UNITED STATES TEL: 715-876-2515 ATTN: TEL: 715-876-2515",
    "MENARDS 7422 EAST STREET Shelby, IA 51570",
]
origin_queries = [
    "TOPOCEAN CONSOLIDATION SERVICE (LAX) INC. 2727 WORKMAN MILL ROAD, CITY OF INDUSTRY, CA 90601 USA TEL: (562)-908-1688 FAX: (562)-908-1699",
    "UNION PACIFIC RR 9TH & JACKSON STREETS OMAHA, NE, 68102 United States FIRMS Code: H043 Port Code: 3512",
    "BNSF - Logistics Park Kansas City - Firm code: J177",
    "UP OMAHA H043 TEL :712-325-6745",
    " CN (Detroit Inter Term) 600 Fern Street Ferndale MI 48220",
    "10201 NW 112th AVENUE, Suite 20 MIAMI, FL, USA NVOCC: 019997N, 33178 Tel.: (305) 863-0007 Fax: (305) 805-8016",
    "DETROIT BRANCH 10725 HARRISON ROAD SUITE 250 ROMULUS, MI 48174",
    "NORFOLK SOUTHERN PIGGYBACK 2725 LIVERNOIS AVE DETROIT MI 48209-1229 UNITED STATES",
    "CALHOUN TRUCK LINES LLC 9845 W 74TH ST EDEN PRAIRIE, MN 55344",
    "Expeditors Intl Of WA Inc 10749 NW Ambassador Drive Kansas City MO 64153 816 880 0900",
    "RADIANT OVERSEAS EXPRESS,2705 S. DIAMOND BAR BLVD. #200 DIAMOND BAR, CA 91765 TEL: (909)468-1969 FAX: (909)468-1985",
    "Kansas City, KS - (32880 West 191 Street,Edgerton KS) - BNSF KANSAS CITY (404)",
    "KINTETSU WORLD EXPRESS 1221 N MITTEL BLVD., WOOD DALE IL 60191 ",
    "9705 NW 108th Ave, Suite 18, Miami, FL 33178. UNITED STATES, Tel: 3055992115, Fax: 8663054950",
    "1701 PIERCE BUTLER RTE SAINT PAUL MN 55104 United States",
    "W3167 County Rd S Iron Ridge WI 53035 United States",
    "CP RAIL-J554",
    "BNSF Railyard/ St Paul Intermodal Facility,1701 Pierce Butler Route,St Paul, Minneapolis, MN",
    "BNSF Railyard/ St Paul Intermodal Facility,1701 Pierce Butler Route,St Paul, Minneapolis, MN",
    "Niagrand, Inc., PO Box 16188, St Paul, MN 55116 Loading at: Salem Tractor Parts 25325 US Hwy 81, Salem, SD 57058605-425-3007 (John Starner)",
    "Dyna Logistics Inc. Tel:310-632-4488 Fax:310-632-0838 SUSANNA_S(susanna@dynalax.com) 19516 S. Susana Rd. ,Rancho Dominguez, CA 90221,",
    "UNION PACIFIC GLOBAL 43000 PATTERSON RD FIRMS CODE I206 JOLIET, IL, 60436, US Contact: UNION PACIFIC RAILROAD COMPANY",
    "PLASTIC RECYCLING, 10252 HIGHWAY 65, IOWA FALLS IA 50126, U.S.A, KEITH ADKINS 509-276-5424",
    "B2B LOGISTICS GROUP INC 500 W. 190TH STREET, SUITE 200 GARDENA, CA 90248 310-618-3700 Phone 310-618-3712 Fax",
    "CSX RAIL ROAD",
    "NORFOLK SOUTHERN - COLUMBUS RICKENBACKER INTERMODAL FACILITY3329 THOROUGHBRED DR COLUMBUS OH 43217 USA TEL:(614) 492-4808",
    "NS COLUMBUS ( H367 ) RICKENBACKER INTERMODAL 3329 THOROUGHBRED DRIVE COLUMBUS, OH 43217 TEL. 614-492-4808",
    "DETROIT INTERMODAL TERMINAL - FIRMS CODE: H798",
    "545 Dowd Avenue Elizabeth, NJ 07201 TELE: 908-345-0555",
    "BNSF LOGISTICS PARK/ J177",
    "CP Minneapolis 615 30th Ave NE MINNEAPOLIS, MN 55418 United States",
    "JIANGSU ZHENGYOUNG FLOORING DECORAT NO.32 CUIBEI VILLAGE , HENGLIN TOWN JIANGSU 213103 CHINA",
    "CSX INTERMODAL, INC 2351 WESTBELT DR COLUMBUS, OH, 43228-3823, US eMail: GO_INTERMODAL@CSX.COM Tel: 800-542-2754 Contact: CSX INTERMODAL, INC",
    "BNSF KANSAS CITY(FIRMS: J177) LOGISTIC PARK 32880 W. 191ST STREET USA TEL:888 428 2673",
    "KANSAS EROSION PRODUCT 3600 AIRPORT RD.,SALINA,KS 67401 SALINA, KANSAS U.S.A",
    "Kansas City Southern RY 4747 Front Street Attn Brian Kleinsorge Kansas City, MO 64120 United States",
    "Kintetsu World Express 235 Southfield Parkway, Suite 100, Forest Park, Georgia 30297 - Phone: 470-373-7200 Fax: 404-366-3749",
    "UP Council Bluffs (IAIS) 2722 South Ave COUNCIL BLUFFS, IA 51501 United States",
    "NYK DELPHINUS 074E, BNSF KANSAS CITY (LPKC) -J177, TOKYO, JAPAN",
    "EAGLEWINGS FREIGHT SERVICES INC. ONE CROSS ISLAND PLAZA, SUITE#312 ROSEDALE, NY 11422 (TEL)718-978 8688 (FAX)718-978 8168",
    "CSX (DETROIT) FIRMS: H797 T: 313-842-2191/297-5440 ",
    "Detroit, MI - (6750 Dix Avenue, Detroit MI) -CSX Detroit (404)",
    "Detroit, MI - (6750 Dix Avenue, Detroit MI) -CSX Detroit (404)",
    "UNION PACIFIC RR 9TH & JACKSON STREETS OMAHA, NE 68102 TEL : FAX : Firm Code : H043",
    "Kintetsu World Express 235 Southfield Parkway, Suite 100, Forest Park, Georgia 30297",
    "GRAND TRUNK RR CONTAINER STA, 600 FERN ST U.S.A., FERNDALE MI 48220, U.S.A.",
    "CP RAIL 615 30TH AVE NE MINNEAPOLIS, MN",
    "MINNEAPOLIS- BNSF",
    "Alliance International 6615 E. Pacific Coast Hwy Suite 100 Long Beach, CA. 90803",
    "CP RAIL - MINNEAPOLIS 615 30TH AVE NE MINNEAPOLIS MN 55418 UNITED STATES",
    "CP RAIL 615 30TH AVE NE MINNEAPOLIS, MN",
    "BNSF RAILWAY 1701 PIERCE BUTLER RTE SAINT PAUL, MN 55104 ,UNITED STATES OF AMERICA",
    "NORFOLK SOUTHERN (NS) - RICKENBACKER INTERMODAL FACILITY H367 3329 THOROUGHBRED DR COLUMBUS OH 43217",
    "Kansas City, KS - (32880 West 191 Street, Edgerton KS) - BNSF KANSAS CITY (404)",
    "NORFOLK SOUTHERN RAMP, 4800 N KIMBALL DR, KANSAS CITY MO 64161, UNITED STATES",
    "PONY EXPRESS WAREHOUSE // ARYSTA LIFESCIENCE, 2307 ALABAMA ST, SAINT JOSEPH MO 64504, UNITED STATES",
    "HAPAG-LLOYD (AMERICA), INC, CANADIAN PACIFIC MINNEAPOLIS 615 30TH AVE., NE MINNEAPOLIS, MN 55418",
    "Allen Lund Company, 475 W. Broadway St, Ste 1, Oviedo FL 32765800-644-5863 Ph 800-479-7771 ",
    "FENGHUA BROTHER WOODEN PRODUCTS COMPANY FENGHUA XIKOU XIUJIABU,ZHEJIANG, CHINA",
    "BNSF Railway Company Firms Code: J177 1275 E 16th Ave Kansas City, MO 64116 United States of America",
    "ALBATRANS INC. 149-10 183rd Street JAMAICA, NY 11413 Tel.+1 718 9176795 Fax:+1 718 9176747",
    "CN-DETROIT (H798) GRW RAIL 600 FERN STREET DETROIT, MI 48220, UNITED STATES TEL: 248-691-6953 ATTN: RAIL TEL: 248-691-6953",
    "QINGDAO YASEN WOOD CO.,LTD #1080 ZHAIZISHAN ROAD, HUANGDAO DISTRICT, QINGDAO CHINA",
    "CANADIAN PACIFIC RAILWAY 615 30TH NE AVE MINNEAPOLIS,MN 55418",
    "GRAND TRUNK RR CONTAINER STA, 600 FERN ST U.S.A., FERNDALE MI 48220 , U.S.A. ",
    "SAVINO DEL BENE U.S.A. INC. 220 West South Thorndale Avenue BENSENVILLE, IL 60106 Tel.847 390 3600 Fax:847 635 6257 E-mail: chicago@savinodelbene.com",
    "BNSF-Omaha 4302 S 8TH ST Omaha, NE 4370 USA",
    "CANADIAN PACIFIC RAILWAY 615 30TH NE AVE MINNEAPOLIS, MN 55418 UNITED STATES OF AMERICA",
    "DETROIT - GTW RR (H798) T: 866-598-8479",
    "CP MINNEAPOLIS (J554) 615 30TH AVENUE N.E MINNEAPOLIS, MN 55418, UNITED STATES",
    "Kansas City, KS - (32880 West 191 Street, Edgerton KS) - BNSF KANSAS CITY (404)",
]
return_queries = [
    "CSX - KANSAS CITY, MO (404) 3301 e 147th st, Kansas City",
    "UNIVERSAL INTERMODAL SERVICES (OMAHA) COUNCIL BLUFFS, IA 51501 TEL: 712-323-9672 ",
    "BNSF Railway Company 1275 E 16th Ave Kansas City, MO 64116 United States of America",
]
data=deliveryAddress.address.to_list()
t=time.time()
# count = 1
for query in data:
    # start = time.time()
    category, output = bm25_deliveryAddress.search(query)
    # end = time.time()
    # print(end-start)
    # print("Query" + str(count) + " ==>" + query)
    # print("Output")
    # print("-" * 50)
    # print("Category == >" + category)
    # print("Result  == >" + output)
    # print("-" * 50)
    # print()
    # print()
    # count += 1
print('totaltime: {}'.format(time.time()-t))
