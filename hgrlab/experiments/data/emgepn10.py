def get_dataset_assets(dataset_type):
    if dataset_type == 'training':
        return TRAINING_DATASET_ASSETS
    elif dataset_type == 'test':
        return TEST_DATASET_ASSETS
    
    return {}

def get_feature_assets(dataset_type):
    if dataset_type == 'training':
        return TRAINING_FEATURE_ASSETS
    elif dataset_type == 'test':
        return TEST_FEATURE_ASSETS
    
    return {}

TRAINING_DATASET_ASSETS = {
    'subject1_training': {
        'remote_id': '1-F1p0N5x2C3fey9GupDYlmHob0l_ZIMg',
        'filename': 'subject1_training.h5',
    },
    'subject2_training': {
        'remote_id': '1-Gjn9KKoFTVH7_O-9L4wOyl2UzX3mIj6',
        'filename': 'subject2_training.h5',
    },
    'subject3_training': {
        'remote_id': '1-r5A0aL4hI9rMsLLGO5O5rgjriTrLMUM',
        'filename': 'subject3_training.h5',
    },
    'subject4_training': {
        'remote_id': '1-8BmKmR4FTVLc8cggZkouCO7BIRHqFuK',
        'filename': 'subject4_training.h5',
    },
    'subject5_training': {
        'remote_id': '1-f0m6uqj3AFMbzA5769TBaEFZDfh6WGa',
        'filename': 'subject5_training.h5',
    },
    'subject6_training': {
        'remote_id': '10BF3MuHphVVq1Mpn-YJ13--VvyQR5HuF',
        'filename': 'subject6_training.h5',
    },
    'subject7_training': {
        'remote_id': '1--vbiMvJL9pvUXOmAr9efVM9iPq-UYog',
        'filename': 'subject7_training.h5',
    },
    'subject8_training': {
        'remote_id': '1-tQbeFzUFqkoun5yXWBvtgYrRycPzuTd',
        'filename': 'subject8_training.h5',
    },
    'subject9_training': {
        'remote_id': '103OiJYh-b6gWv5Yew3JTwFo2S7LgbmE7',
        'filename': 'subject9_training.h5',
    },
    'subject10_training': {
        'remote_id': '1-3aMg8xDAVBknSLWnNbNtX8dHiNjZofe',
        'filename': 'subject10_training.h5',
    }
}

TEST_DATASET_ASSETS = {}

TRAINING_FEATURE_ASSETS = {}

TEST_FEATURE_ASSETS = {
    'subject1_test_features_1': {
        'remote_id': '1rLtTvjUC-Ms6HqZJsstIQOB3ZR4hcoi8',
        'filename': 'emgepn30_features_subject1_test_1b57d5b091d4ad3a12ef5ecd749dc5e7.h5',
    },
    'subject1_test_features_2': {
        'remote_id': '1rtASIrqwxy9vqmuEWBbJRCor3tPYUWKl',
        'filename': 'emgepn30_features_subject1_test_275faa72d1fde5ff75d575839e2fc4b4.h5',
    },
    'subject2_test_features_1': {
        'remote_id': '1rDxrwPqV8JrGkAGThfSPMW4F3S5lerWF',
        'filename': 'emgepn30_features_subject2_test_bc6ff8cbac25292050958fd345ef3fa8.h5',
    },
    'subject2_test_features_2': {
        'remote_id': '1sXdhyWOUNqtbXgX6_Dw4XaaGRhbQ6kRW',
        'filename': 'emgepn30_features_subject2_test_c450226bdfd56d2a447763c8b143e32e.h5',
    },
    'subject3_test_features_1': {
        'remote_id': '1sPIk5jzGWuop7F6tBMW2a94mMoeapZB7',
        'filename': 'emgepn30_features_subject3_test_622e160136988940243023c1c2eca08e.h5',
    },
    'subject3_test_features_2': {
        'remote_id': '1rObfE33ZvrojGMGyLd4EdTH8ykLltYys',
        'filename': 'emgepn30_features_subject3_test_7602c145973b875375978741675d200f.h5',
    },
    'subject3_test_features_3': {
        'remote_id': '1rWPPYvIxR0_RBacS8buw-EIt8N3VyyfE',
        'filename': 'emgepn30_features_subject3_test_bdced377ed1b9119ede3ac699b24c7d5.h5',
    },
    'subject3_test_features_4': {
        'remote_id': '1rtUwTRMUz1X7L_cUotdAJq6IkCgUyQX6',
        'filename': 'emgepn30_features_subject3_test_d6a49b2e85d2f09ca6d215806a985432.h5',
    },
    'subject4_test_features_1': {
        'remote_id': '1tVkftGuh6oTFBMS8G-QJ7aA8mqJh0QTQ',
        'filename': 'emgepn30_features_subject4_test_9f6c2bc1afc7c34f90e412976c1299d1.h5',
    },
    'subject4_test_features_2': {
        'remote_id': '1sKiuZSxynHbD5lQJT3R2-XEMoazhkFTk',
        'filename': 'emgepn30_features_subject4_test_dc5126d46ccf581299e73d9286364873.h5',
    },
    'subject5_test_features_1': {
        'remote_id': '1rvq3st8Qm_AtleCF9HuCm31LUso79axh',
        'filename': 'emgepn30_features_subject5_test_052ead8353b5136d5b4b1ae407437ee4.h5',
    },
    'subject5_test_features_2': {
        'remote_id': '1t2EqKbequhVMXkgrPv2X-SF6IuS1jFcr',
        'filename': 'emgepn30_features_subject5_test_96157d0333c8b5523942efd86d1627ef.h5',
    },
    'subject5_test_features_3': {
        'remote_id': '1rF6afXyn-eRWer1LiH00krg0o1Soj2EO',
        'filename': 'emgepn30_features_subject5_test_b11c9c2bed8ec31d7cbf9e76436466f5.h5',
    },
    'subject5_test_features_4': {
        'remote_id': '1sz0bk1ueV9SzRh9CD7nYNilNwLKBHuAS',
        'filename': 'emgepn30_features_subject5_test_d75f0326c7b26ee2cc9bcb97694373e7.h5',
    },
    'subject6_test_features_1': {
        'remote_id': '1sJFQiEJ4mK7L5XJlO2UkRGBu0eC6B6D-',
        'filename': 'emgepn30_features_subject6_test_7da80703509bc75a0dd01d811427d9c2.h5',
    },
    'subject6_test_features_2': {
        'remote_id': '1rEqgCLf6aev-XNLfz9RCkAxB2TDPmns4',
        'filename': 'emgepn30_features_subject6_test_69d0e5db3aa966f6b368db5ade9d4774.h5',
    },
    'subject6_test_features_3': {
        'remote_id': '1s0286jq8AgP-_IwHRcKJdJtFZOC_nYdL',
        'filename': 'emgepn30_features_subject6_test_4304c3177801b2c6f288a2e92f90c2aa.h5',
    },
    'subject7_test_features_1': {
        'remote_id': '1ro0uJ0xObPdbJjnBSMANtTPKJyzXtbgp',
        'filename': 'emgepn30_features_subject7_test_51dd2cd2ba70dcc68288bbce53f56670.h5',
    },
    'subject7_test_features_2': {
        'remote_id': '1sI395UW55qEQiEz1KvGVPCa8DLzIR7RI',
        'filename': 'emgepn30_features_subject7_test_d1bcff6dca1f9f4753da18961ce384a3.h5',
    },
    'subject8_test_features_1': {
        'remote_id': '1tsizP4gw0upYnKqOmX0-eiYN3Mmigl74',
        'filename': 'emgepn30_features_subject8_test_78c926c1455722501ad0bc662a2cff11.h5',
    },
    'subject9_test_features_1': {
        'remote_id': '1roMDn9gHZbhzBWhvPdIwlrdqDunjiZqd',
        'filename': 'emgepn30_features_subject9_test_b627bacddbdd8ce47b7a289fa05b2a03.h5',
    },
    'subject10_test_features_1': {
        'remote_id': '1sJHSYMa3zW4Eao9tbWbI2tlWSCIX0aZj',
        'filename': 'emgepn30_features_subject10_test_55fb5ad833c5dab92e8436482a6460bb.h5',
    },
    'subject10_test_features_2': {
        'remote_id': '1tcipgt0GXLWDyTfveeO40VzCqc9mZuhC',
        'filename': 'emgepn30_features_subject10_test_637970283bcf00faf5b7c17542794b4d.h5',
    },
}
