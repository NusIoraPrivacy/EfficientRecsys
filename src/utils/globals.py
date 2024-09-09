
norm_dict = {"MF": 9, "NCF": (6, 6)}

distribution_dict = {"MF": (0.3, 1.5), "NCF": (0.01, 1.2)}

rating_range = {"ml-1m": (1, 5)}

private_param_dict = {
                    "MF": ["embedding_user.weight"], 
                    "NCF": ["gmf_embedding_user.weight", "ncf_embedding_user.weight"]
                    }