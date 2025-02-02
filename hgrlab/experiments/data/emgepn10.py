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

TEST_DATASET_ASSETS = {
    'subject1_test': {
        'remote_id': '1-EYC55D9nE7asCM3mSiCUqaC8JQzEuj3',
        'filename': 'subject1_test.h5',
    },
    'subject2_test': {
        'remote_id': '1-8fO2054FtcgHOqhs0_1kZCQyQ-tPjRI',
        'filename': 'subject2_test.h5',
    },
    'subject3_test': {
        'remote_id': '1-5jKP3nyzOQUJBsM50gVjrHL33enPqdA',
        'filename': 'subject3_test.h5',
    },
    'subject4_test': {
        'remote_id': '1-iPHHhFHM28M9Vr7mSlsKZXfyqjNXBeT',
        'filename': 'subject4_test.h5',
    },
    'subject5_test': {
        'remote_id': '1-7mKYlqodvTGpARkmhzPiXIwQk9kG0M2',
        'filename': 'subject5_test.h5',
    },
    'subject6_test': {
        'remote_id': '1-a0sQQ0Tr_Byhtlh_rB8KJkD7w7bDSK6',
        'filename': 'subject6_test.h5',
    },
    'subject7_test': {
        'remote_id': '1--eeuTAcP43MPwRR7roHQbLPFAlbVcl3',
        'filename': 'subject7_test.h5',
    },
    'subject8_test': {
        'remote_id': '105slJnmQdTTrCV7IStCExvX7wiakzAsb',
        'filename': 'subject8_test.h5',
    },
    'subject9_test': {
        'remote_id': '1-r_aTArNhOUVpd_avgz8F2ORgpwKnZBK',
        'filename': 'subject9_test.h5',
    },
    'subject10_test': {
        'remote_id': '1-332PJLsWOFAxIq9S7P93PW6xvMMF1gh',
        'filename': 'subject10_test.h5',
    },
}

TRAINING_FEATURE_ASSETS = {}

TEST_FEATURE_ASSETS = {
    'features_subject1_test_fastdtw_t18': {
        'remote_id': '13u5RVl-hxZQWMeiTtQJ9dVlxwyThx2EJ',
        'filename': 'emgepn10_features_subject1_test_fastdtw_a5a723dd40925ba75b4cbe7ab0f9e460.h5',
    },
    'features_subject1_test_fastdtw_t19': {
        'remote_id': '1rtASIrqwxy9vqmuEWBbJRCor3tPYUWKl',
        'filename': 'emgepn10_features_subject1_test_fastdtw_da1d77954164b40f4be4fbeceba2fad2.h5',
    },
    'features_subject2_test_fastdtw_t18': {
        'remote_id': '1sXdhyWOUNqtbXgX6_Dw4XaaGRhbQ6kRW',
        'filename': 'emgepn10_features_subject2_test_fastdtw_3a748e61dc1351e0356314bef261dd61.h5',
    },
    'features_subject2_test_fastdtw_t20': {
        'remote_id': '1rDxrwPqV8JrGkAGThfSPMW4F3S5lerWF',
        'filename': 'emgepn10_features_subject2_test_fastdtw_ef7a2e96ab67adb9c797d10a42ec7719.h5',
    },
    'features_subject3_test_fastdtw_t12': {
        'remote_id': '1sPIk5jzGWuop7F6tBMW2a94mMoeapZB7',
        'filename': 'emgepn10_features_subject3_test_fastdtw_d90af7859dc9b3570691f5345a5a7a7b.h5',
    },
    'features_subject3_test_fastdtw_t15': {
        'remote_id': '1rtUwTRMUz1X7L_cUotdAJq6IkCgUyQX6',
        'filename': 'emgepn10_features_subject3_test_fastdtw_0ab02d9c3f63a7ad65235e6fd99423dc.h5',
    },
    'features_subject3_test_fastdtw_t19': {
        'remote_id': '1rObfE33ZvrojGMGyLd4EdTH8ykLltYys',
        'filename': 'emgepn10_features_subject3_test_fastdtw_5c8b126a694e4c1de508386af2bc0192.h5',
    },
    'features_subject3_test_fastdtw_t20': {
        'remote_id': '1rWPPYvIxR0_RBacS8buw-EIt8N3VyyfE',
        'filename': 'emgepn10_features_subject3_test_fastdtw_432af5f839d9a7800c2adbcd7dbe910d.h5',
    },
    'features_subject4_test_fastdtw_t16': {
        'remote_id': '1sKiuZSxynHbD5lQJT3R2-XEMoazhkFTk',
        'filename': 'emgepn10_features_subject4_test_fastdtw_5e58fa53a121931f58671ad5b412e951.h5',
    },
    'features_subject4_test_fastdtw_t19': {
        'remote_id': '1tVkftGuh6oTFBMS8G-QJ7aA8mqJh0QTQ',
        'filename': 'emgepn10_features_subject4_test_fastdtw_2fdcb13024abbae1f52c95683b4ab557.h5',
    },
    'features_subject5_test_fastdtw_t11': {
        'remote_id': '1t2EqKbequhVMXkgrPv2X-SF6IuS1jFcr',
        'filename': 'emgepn10_features_subject5_test_fastdtw_91233087d802474214b5e4e60f7b38e5.h5',
    },
    'features_subject5_test_fastdtw_t14': {
        'remote_id': '1rvq3st8Qm_AtleCF9HuCm31LUso79axh',
        'filename': 'emgepn10_features_subject5_test_fastdtw_d75136344b3d758d406c956e1ff0579d.h5',
    },
    'features_subject5_test_fastdtw_t17': {
        'remote_id': '1sz0bk1ueV9SzRh9CD7nYNilNwLKBHuAS',
        'filename': 'emgepn10_features_subject5_test_fastdtw_131ae299de51ce9b6698f47526c0ea16.h5',
    },
    'features_subject5_test_fastdtw_t19': {
        'remote_id': '1rF6afXyn-eRWer1LiH00krg0o1Soj2EO',
        'filename': 'emgepn10_features_subject5_test_fastdtw_d6f48372ce69df6e212888257f41c905.h5',
    },
    'features_subject6_test_fastdtw_t16': {
        'remote_id': '1sJFQiEJ4mK7L5XJlO2UkRGBu0eC6B6D-',
        'filename': 'emgepn10_features_subject6_test_fastdtw_830e077b906b41b9ddda7fbb8197e87a.h5',
    },
    'features_subject6_test_fastdtw_t18': {
        'remote_id': '1s0286jq8AgP-_IwHRcKJdJtFZOC_nYdL',
        'filename': 'emgepn10_features_subject6_test_fastdtw_8bd090ab9b3914d87086825b444fa4fc.h5',
    },
    'features_subject6_test_fastdtw_t19': {
        'remote_id': '1rEqgCLf6aev-XNLfz9RCkAxB2TDPmns4',
        'filename': 'emgepn10_features_subject6_test_fastdtw_02622fe862f48128151878449b955d6e.h5',
    },
    'features_subject7_test_fastdtw__t14': {
        'remote_id': '1ro0uJ0xObPdbJjnBSMANtTPKJyzXtbgp',
        'filename': 'emgepn10_features_subject7_test_fastdtw_5ee0dba507588507b19c20e701c5c4e6.h5',
    },
    'features_subject7_test_fastdtw__t19': {
        'remote_id': '1sI395UW55qEQiEz1KvGVPCa8DLzIR7RI',
        'filename': 'emgepn10_features_subject7_test_fastdtw_65a0a3e1694d0f5ccaa790232db1d0f0.h5',
    },
    'features_subject8_test_fastdtw_t19': {
        'remote_id': '1tsizP4gw0upYnKqOmX0-eiYN3Mmigl74',
        'filename': 'emgepn10_features_subject8_test_fastdtw_2df5c35821feb5fbcd2017d6f63b1f36.h5',
    },
    'features_subject9_test_fastdtw_t19': {
        'remote_id': '1roMDn9gHZbhzBWhvPdIwlrdqDunjiZqd',
        'filename': 'emgepn10_features_subject9_test_fastdtw_a5349d2833e269573a4db94f08a5ebc8.h5',
    },
    'features_subject10_test_fastdtw__t19': {
        'remote_id': '1sJHSYMa3zW4Eao9tbWbI2tlWSCIX0aZj',
        'filename': 'emgepn10_features_subject10_test_fastdtw_6fad6985589155a1c8621376840734df.h5',
    },
    'features_subject10_test_fastdtw_t20': {
        'remote_id': '1tcipgt0GXLWDyTfveeO40VzCqc9mZuhC',
        'filename': 'emgepn10_features_subject10_test_fastdtw_512ab6fd23581888fff7d140ed2a48b3.h5',
    },
}
