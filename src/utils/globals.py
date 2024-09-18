
norm_dict = {"ml-1m": {"MF": 9, "NCF": (3.5, 2.8), "FM": (5.5, 2), "DeepFM": (4.5, 2.7)},
            "ml-10m": {"MF": 9, "NCF": (3.5, 2.8), "FM": (5.5, 2), "DeepFM": (4.5, 2.7)},
            "bookcrossing": {"MF": 9, "NCF": (3.5, 2.8), "FM": (5.5, 2), "DeepFM": (4.5, 2.7)},
            "yelp": {"MF": 1.4, "NCF": (0.12, 0.10), "FM": (0.32, 0), "DeepFM": (0.35, 0)},
            }

distribution_dict = {"MF": (0.3, 1.5), "NCF": (0.01, 1.2)}

rating_range = {"ml-1m": (1, 5), "bookcrossing": (0, 10), "yelp": (1, 5), "ml-100k": (1, 5), "ml-10m": (1, 5)}

# rating_thds = {"bookcrossing": (5, 10)} # (15, 15) 10391 items + 4517 users average 69 rated per user
rating_thds = {"bookcrossing": (15, 15)} # (15, 15) 10391 items + 4517 users average 69 rated per user

private_param_dict = {
                    "MF": ["embedding_user.weight"], 
                    "NCF": ["gmf_embedding_user.weight", "ncf_embedding_user.weight"],
                    "FM": ["embedding_user.weight"],
                    "DeepFM": ["embedding_user.weight"],
                    }

all_genre_dict = {
    "ml-1m": ['Animation', "Children's", 'Comedy', 'Adventure', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'Sci-Fi', 'Documentary', 'War', 'Musical', 'Mystery', 'Film-Noir', 'Western'],
    "ml-10m": ['Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'Mystery', 'Sci-Fi', 'IMAX', 'Documentary', 'War', 'Musical', 'Film-Noir', 'Western', '(no genres listed)']
}

models_w_feats = ["FM", "DeepFM"]

n_max_user_feat_dict = {
    "ml-1m": 3,
    "ml-100k": 3,
    "yelp": 0,
    "ml-10m": 0,
}

replace_loc = {
    'united kingdom.': 'united kingdom',
    "lj": "na",
    "quit": "na",
    '"': "na",
    'the': "na",
    "n/a": "na",
    "csa": "na",
    'l`italia': "italy",
    'la france': 'france',
    "19104": "na",
    "england": "united kingdom",
    'england uk': "united kingdom",
    "öð¹ú": "na",
    "&#32654;&#22269;": "na",
    'u.s. of a.': "usa",
    'serbia & montenegro': "serbia and montenegro",
    '-------': "na",
    'pakistan.': 'pakistan',
    'trinidad/tobago.': 'trinidad and tobago',
    'tobago': 'trinidad and tobago',
    'trinidad & tobago': 'trinidad and tobago',
    'sri lanka\\"n/a': "sri lanka",
    'catalunya(catalonia)': "catalonia",
    "catalunya": "catalonia",
    'catalunya spain': "catalonia",
    "doodedoo": "na",
    'de': 'germany',
    '': "na",
    'uk': "united kingdom",
    'mã?â©xico': "mexico",
    'tdzimi': "na",
    "itlay": "italy",
    'u.s.a.': "usa",
    'st. vincent and the grenadines': 'saint vincent and the grenadines',
    'zhengjiang': "china",
    'p.r.china': "china",
    'china öð¹ú': "china",
    'pr': 'puerto rico',
    'good old usa !': "usa",
    'bbbzzzzz': "na",
    'fiji': "na",
    'lake': "usa",
    '*': "na",
    'united stated': "usa",
    "the great white north": "na",
    'lkjlj': "na",
    "?ú?{' 'c": "na",
    "cnina": "china",
    "espaã?â±a": "spain",
    "la belgique": "belgium",
    'n/a - on the road': "na",
    "ua": "usa",
    'sri lanka\\"n/a': "sri lanka",
    'srilanka': "sri lanka",
    "republic of korea": 'south korea',
    'côte d': 'cote d`ivoire',
    "saint loius": "usa",
    'saint luica': 'saint lucia',
    'united kindgonm': "united kingdom",
    'nyc': "usa",
    'san bernardino': "usa",
    'ee.uu': "usa",
    'here and there': "na",
    'espaã±a': "spain",
    "channel islands": "united kingdom",
    '87510': "na",
    'ä¸\xadå?½': "na",
    'méxico': "mexico",
    'usa & canada': "usa",
    '"n/a': "na",
    'thing': "na",
    'we`re global!': "na",
    'europe.': "europe",
    'the philippines': 'philippines',
    'philippine': 'philippines',
    'philippinies': "philippines",
    'u.k.': "united kingdom",
    "santa barbara": "usa",
    'u.s.a>': "usa",
    'u k': "united kingdom",
    'europa': "europe",
    'malaysian': 'malaysia',
    'w. malaysia': 'malaysia',
    "usa (currently living in england)": "usa",
    "p.r. china": "china",
    'brasil': "brazil",
    '_ brasil': "brazil",
    'united kindgdom': "united kingdom",
    'il canada': "canada",
    'cananda': "canada",
    'united state': "usa",
    'unite states': "usa",
    'republic of panama': "panama",
    'türkiye': 'turkey',
    'california': "usa",
    "people`s republic of china": "china",
    'geermany': "germany",
    'germay': "germany",
    "p r china": "china",
    'victoria': 'australia',
    'la argentina': 'argentina',
    "españa": "spain",
    "la suisse": 'switzerland',
    "fernando de la mora": 'paraguay',
    'america': "usa",
    'u.s.a!': "usa",
    'u.s>': "usa",
    'san franicsco': "usa",
    "greece (=hellas)": "greece",
    'us': "usa",
    "pa": "panama",
    'united states of america': "usa",
    'singapore/united kingdom': "singapore",
    'la svizzera': 'switzerland',
    'chinaöð¹ú': "china",
    'united sates': "usa",
    'st.thomasi': 'u.s. virgin islands',
    'united states': "usa",
    'united statea': "usa",
    "perãº": "peru",
    "oakland": "usa",
    "c.a.": "belize",
    "swazilandia": 'swaziland',
    "p.r.c": "china",
    "los estados unidos de norte america": "usa",
    "slovak republik": "slovakia",
    "antigua & barbuda": "antigua and barbuda",
    "algérie": "algeria",
    'new london': "usa",
    "italia": "italy",
    'u.a.e': 'united arab emirates',
    'brunei darussalam': "brunei",
    "holy see": "italy",
    "the netherlands": 'netherlands',
    "le canada": "canada",
    "polk": "usa"
}

common_countries = ['usa', 'canada', 'united kingdom', 'germany', 'spain', 'australia', 'italy', 'na', 'france', 'portugal', 'new zealand', 'netherlands', 'switzerland', 'brazil', 'china', 'sweden', 'india', 'austria', 'malaysia', 'argentina', 'finland', 'singapore', 'denmark', 'mexico', 'belgium', 'ireland', 'philippines', 'turkey', 'poland', 'pakistan', 'greece', 'iran', 'romania', 'chile', 'israel', 'south africa', 'indonesia', 'norway', 'japan', 'croatia', 'nigeria', 'south korea', 'slovakia', 'czech republic', 'russia', 'yugoslavia', 'hong kong', 'costa rica', 'taiwan', 'slovenia', 'egypt', 'peru', 'vietnam', 'venezuela', 'colombia', 'bulgaria', 'luxembourg', 'hungary', 'thailand', 'united arab emirates', 'ghana', 'saudi arabia', 'sri lanka', 'bosnia and herzegovina', 'iceland', 'bangladesh', 'paraguay', 'guatemala', 'andorra', 'lithuania', 'ukraine', 'bahamas', 'latvia', 'bolivia', 'panama', 'trinidad and tobago', 'jamaica', 'ecuador', 'kuwait', 'lebanon', 'cuba', 'morocco', 'malta', 'scotland', 'macedonia', 'afghanistan', 'albania', 'dominican republic', 'algeria', 'urugua', 'deutschland', 'honduras', 'antarctica', 'cyprus', 'kenya', 'bermuda', 'el salvador', 'oman', 'belize', 'catalonia', 'uzbekistan', 'zimbabwe', 'estonia', 'jordan', 'puerto rico', 'mauritius', 'nepal', 'uruguay', 'brunei', 'grenada', 'qatar', 'burma', 'barbados', 'nicaragua', 'wales', 'caribbean sea', 'bahrain', 'iraq', 'jersey', 'georgia', 'belarus', 'benin', 'east africa', 'mozambique', 'kazakhstan', 'syria', 'ethiopia', 'cote d`ivoire', 'cayman islands', 'euskal herria', 'sudan', 'guernsey', 'alderney', 'guyana', 'antigua and barbuda', 'eritrea', 'azerbaijan', 'armenia', 'uganda', 'cape verde', 'moldova', 'tunisia', 'monaco', 'yemen', 'cameroon', 'papua new guinea', 'gabon', 'botswana', 'dominica', 'galiza', 'laos', 'saint vincent and the grenadines', 'tanzania', 'cambodia', 'rwanda', 'togo', 'niger', 'saint lucia', 'netherlands antilles', 'burkina faso', 'zambia', 'samoa', 'senegal', 'angola', 'europe', 'north korea', 'kyrgyzstan', 'congo', 'trinidad', 'maricopa', 'aruba', 'basque country', 'mongolia', 'lombardia', 'suriname', 'libya', 'suisse', 'lesotho', 'korea', 'vanuatu', 'bhutan', 'kosovo', 'belgique', 'macau', 'namibia', 'maldives', 'lazio', 'swaziland', 'serbia', 'tajikistan', 'san marino', 'u.s. virgin islands', 'serbia and montenegro', 'chad', 'guinea-bissau', 'liberia', 'solomon islands', 'palau']