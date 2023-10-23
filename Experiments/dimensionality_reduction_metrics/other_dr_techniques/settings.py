SETTINGS= {
    "UMap": {
                "def": {"angular_rp_forest": False, "init": "spectral",  "min_dist": 0.001, "n_neighbors": 5, "spread": 1.0},

                "setting1":{'angular_rp_forest': False, 'init': 'spectral',  'min_dist': 0.001,  'n_neighbors': 6, 'spread': 1.0},

                "setting2":{'angular_rp_forest': False, 'init': 'spectral',  'min_dist': 0.001,  'n_neighbors': 7,  'spread': 1.0},

                "setting3":{'angular_rp_forest': False, 'init': 'spectral',  'min_dist': 0.001,  'n_neighbors': 8,  'spread': 1.0},

                "setting4":{'angular_rp_forest': False, 'init': 'spectral',  'min_dist': 0.001,  'n_neighbors': 4,  'spread': 1.0},

                "setting5": {'angular_rp_forest': False, 'init': 'spectral',  'min_dist': 0.001,  'n_neighbors': 3,  'spread': 1.0},

                "setting6":{'angular_rp_forest': False, 'init': 'spectral',  'min_dist': 0.0005,  'n_neighbors': 5,  'spread': 1.0},

                "setting7": {'angular_rp_forest': False, 'init': 'spectral',  'min_dist': 0.002,  'n_neighbors': 5,  'spread': 1.0},

                "setting8":{'angular_rp_forest': False, 'init': 'random',  'min_dist': 0.001,  'n_neighbors': 5,  'spread': 1.0, 'random_state': 25 }

            },
    "UMap2": {
                "def": {"angular_rp_forest": False, "init": "spectral",  "min_dist": 0.001, "n_neighbors": 5, "spread": 1.0},

                "setting1":{'angular_rp_forest': False, 'init': 'spectral',  'min_dist': 0.001,  'n_neighbors': 6, 'spread': 1.0},

                "setting2":{'angular_rp_forest': False, 'init': 'spectral',  'min_dist': 0.001,  'n_neighbors': 7,  'spread': 1.0},

                "setting3":{'angular_rp_forest': False, 'init': 'spectral',  'min_dist': 0.001,  'n_neighbors': 8,  'spread': 1.0},

                "setting4":{'angular_rp_forest': False, 'init': 'spectral',  'min_dist': 0.001,  'n_neighbors': 4,  'spread': 1.0},

                "setting5": {'angular_rp_forest': False, 'init': 'spectral',  'min_dist': 0.001,  'n_neighbors': 3,  'spread': 1.0},

                "setting6":{'angular_rp_forest': False, 'init': 'spectral',  'min_dist': 0.0005,  'n_neighbors': 5,  'spread': 1.0},

                "setting7": {'angular_rp_forest': False, 'init': 'spectral',  'min_dist': 0.002,  'n_neighbors': 5,  'spread': 1.0},

                "setting8":{'angular_rp_forest': False, 'init': 'random',  'min_dist': 0.001,  'n_neighbors': 5,  'spread': 1.0, 'random_state': 25 }

            },
    
    "T-SNE": {
                "def": {'angle': 0.5, 'early_exaggeration': 12.0, 'init': 'random', 'learning_rate': 200.0, 'method': 'barnes_hut',  'min_grad_norm': 1e-07, 'n_components': 2, 'n_iter': 1000, 'n_iter_without_progress': 300,  'perplexity': 50.0},

                "setting1": {'angle': 0.5, 'early_exaggeration': 10.0, 'init': 'random', 'learning_rate': 200.0, 'method': 'barnes_hut',  'min_grad_norm': 1e-07, 'n_components': 2, 'n_iter': 1000, 'n_iter_without_progress': 300,  'perplexity': 50.0},

                "setting2": {'angle': 0.5, 'early_exaggeration': 14.0, 'init': 'random', 'learning_rate': 200.0, 'method': 'barnes_hut',  'min_grad_norm': 1e-07, 'n_components': 2, 'n_iter': 1000, 'n_iter_without_progress': 300,  'perplexity': 50.0},

                "setting3": {'angle': 0.5, 'early_exaggeration': 12.0, 'init': 'random', 'learning_rate': 250.0, 'method': 'barnes_hut',  'min_grad_norm': 1e-07, 'n_components': 2, 'n_iter': 1000, 'n_iter_without_progress': 300,  'perplexity': 50.0},

                "setting4" : {'angle': 0.5, 'early_exaggeration': 12.0, 'init': 'random', 'learning_rate': 150.0, 'method': 'barnes_hut',  'min_grad_norm': 1e-07, 'n_components': 2, 'n_iter': 1000, 'n_iter_without_progress': 300,  'perplexity': 50.0},

                "setting5" : {'angle': 0.5, 'early_exaggeration': 12.0, 'init': 'pca', 'learning_rate': 200.0, 'method': 'barnes_hut',  'min_grad_norm': 1e-07, 'n_components': 2, 'n_iter': 1000, 'n_iter_without_progress': 300, 'perplexity': 50.0}
    },

    "S-PCA": {
                "def" : {'alpha': 0.01, 'max_iter': 1000, 'method': 'lars', 'normalize_components': True,  'ridge_alpha': 0.05, 'tol': 1e-08},

                "setting1" : {'alpha': 0.005, 'max_iter': 1000, 'method': 'lars', 'normalize_components': True,  'ridge_alpha': 0.05, 'tol': 1e-08},

                "setting2" : {'alpha': 0.001, 'max_iter': 1000, 'method': 'lars', 'normalize_components': True,  'ridge_alpha': 0.05, 'tol': 1e-08}, 

                "setting3" : {'alpha': 0.009, 'max_iter': 1000, 'method': 'lars', 'normalize_components': True,  'ridge_alpha': 0.05, 'tol': 1e-08},

                "setting4": {'alpha': 0.02, 'max_iter': 1000, 'method': 'lars', 'normalize_components': True,  'ridge_alpha': 0.05, 'tol': 1e-08},

                "setting5": {'alpha': 0.03, 'max_iter': 1000, 'method': 'lars', 'normalize_components': True,  'ridge_alpha': 0.05, 'tol': 1e-08},

                "setting6" : {'alpha': 0.01, 'max_iter': 1000, 'method': 'cd', 'normalize_components': True,  'ridge_alpha': 0.05, 'tol': 1e-08}

    },

    "M-TSNE": {
                "def": {'angle': 0.5, 'early_exaggeration': 12.0, 'init': 'random', 'learning_rate': 200.0, 'method': 'barnes_hut',  'min_grad_norm': 1e-07, 'n_components': 2, 'n_iter': 1000, 'n_iter_without_progress': 300,  'perplexity': 50.0},

                "setting1": {'angle': 0.5, 'early_exaggeration': 10.0, 'init': 'random', 'learning_rate': 200.0, 'method': 'barnes_hut',  'min_grad_norm': 1e-07, 'n_components': 2, 'n_iter': 1000, 'n_iter_without_progress': 300,  'perplexity': 50.0},

                "setting2": {'angle': 0.5, 'early_exaggeration': 14.0, 'init': 'random', 'learning_rate': 200.0, 'method': 'barnes_hut',  'min_grad_norm': 1e-07, 'n_components': 2, 'n_iter': 1000, 'n_iter_without_progress': 300,  'perplexity': 50.0},

                "setting3": {'angle': 0.5, 'early_exaggeration': 12.0, 'init': 'random', 'learning_rate': 250.0, 'method': 'barnes_hut',  'min_grad_norm': 1e-07, 'n_components': 2, 'n_iter': 1000, 'n_iter_without_progress': 300,  'perplexity': 50.0},

                "setting4" : {'angle': 0.5, 'early_exaggeration': 12.0, 'init': 'random', 'learning_rate': 150.0, 'method': 'barnes_hut',  'min_grad_norm': 1e-07, 'n_components': 2, 'n_iter': 1000, 'n_iter_without_progress': 300,  'perplexity': 50.0},

                "setting5" : {'angle': 0.5, 'early_exaggeration': 12.0, 'init': 'pca', 'learning_rate': 200.0, 'method': 'barnes_hut',  'min_grad_norm': 1e-07, 'n_components': 2, 'n_iter': 1000, 'n_iter_without_progress': 300, 'perplexity': 50.0}
    },

    "K-PCA" : {

                'def' : {'gamma': None, 'kernel': 'rbf', 'max_iter': None}

    }
}