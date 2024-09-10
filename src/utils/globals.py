
norm_dict = {"MF": 9, "NCF": (3.5, 2.8), "FM": ()}

distribution_dict = {"MF": (0.3, 1.5), "NCF": (0.01, 1.2)}

rating_range = {"ml-1m": (1, 5)}

private_param_dict = {
                    "MF": ["embedding_user.weight"], 
                    "NCF": ["gmf_embedding_user.weight", "ncf_embedding_user.weight"],
                    "FM": ["embedding_user.weight"],
                    }

all_genre_dict = {
    "ml-1m": ['Animation', "Children's", 'Comedy', 'Adventure', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'Sci-Fi', 'Documentary', 'War', 'Musical', 'Mystery', 'Film-Noir', 'Western']
}