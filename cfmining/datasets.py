import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


class Dataset:
    def __init__(self, name):
        self.name = name
        self.outlier_contamination = 0.05
        self.categoric_features = []
        self.not_mutable_features = []

    def __repr__(self):
        return self.name

    def load_data(self):
        self.dataframe = pd.read_csv(self.path)

        X = self.dataframe.drop(columns=[self.target])
        y = self.dataframe[self.target]

        if not self.use_categorical:
            X = X.drop(columns=self.categoric_features)
            self.categoric_features = []
        else:
            # perform Ordinal Encoding
            encoder = OrdinalEncoder(
                dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1
            )
            X[self.categoric_features] = encoder.fit_transform(
                X[self.categoric_features]
            )

        self.mutable_features = X.columns
        self.mutable_features = [
            feat
            for feat in self.mutable_features
            if feat not in self.not_mutable_features
        ]

        self.continuous_features = [
            feat for feat in X.columns if feat not in self.categoric_features
        ]

        return X, y


class GermanCredit(Dataset):
    def __init__(self, use_categorical=False):
        super().__init__("german")
        self.use_categorical = use_categorical
        self.outlier_contamination = 0.05
        self.categoric_features = ["PurposeOfLoan"]
        self.target = "GoodCustomer"
        self.not_mutable_features = [
            "Age",
            "OwnsHouse",
            "isMale",
            "JobClassIsSkilled",
            "Single",
            "ForeignWorker",
            "RentsHouse",
        ]
        self.path = "../data/german.csv"


class Taiwan(Dataset):
    def __init__(self, use_categorical=False):
        super().__init__("taiwan")
        self.use_categorical = use_categorical
        self.outlier_contamination = 0.05
        self.categoric_features = ["EDUCATION", "MARRIAGE"]
        self.target = "NoDefaultNextMonth"
        self.not_mutable_features = ["Age", "MARRIAGE"]
        self.path = "../data/taiwan.csv"


class Adult(Dataset):
    def __init__(self, use_categorical=False):
        super().__init__("adult")
        self.use_categorical = use_categorical
        self.outlier_contamination = 0.05
        self.categoric_features = [
            "occupation",
            "relationship",
            "race",
        ]
        self.target = "income"
        self.not_mutable_features = ["race", "is_male", "age"]
        self.path = "../data/adult.csv"


class ACSIncome(Dataset):
    def __init__(self, use_categorical=False):
        super().__init__("acsincome")
        self.use_categorical = use_categorical
        self.outlier_contamination = 0.05
        self.target = "target"
        self.categoric_features = []
        self.not_mutable_features = ["SEX", "AGEP"]
        self.path = "../data/acsincome.csv"


class HomeCredit(Dataset):
    def __init__(self, use_categorical=False):
        super().__init__("homecredit")
        self.use_categorical = use_categorical
        self.outlier_contamination = 0.05
        self.target = "TARGET"
        self.categoric_features = [
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_HOUSING_TYPE",
            "OCCUPATION_TYPE",
            "NAME_CONTRACT_TYPE",
        ]
        self.not_mutable_features = [
            "NAME_CONTRACT_TYPE",
            "CNT_CHILDREN",
            "NAME_INCOME_TYPE",
            "NAME_EDUCATION_TYPE",
            "NAME_HOUSING_TYPE",
            "REGION_POPULATION_RELATIVE",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "FLAG_MOBIL",
            "FLAG_EMP_PHONE",
            "FLAG_WORK_PHONE",
            "FLAG_CONT_MOBILE",
            "FLAG_PHONE",
            "FLAG_EMAIL",
            "OCCUPATION_TYPE",
            "CNT_FAM_MEMBERS",
            "REGION_RATING_CLIENT",
            "REGION_RATING_CLIENT_W_CITY",
            "REG_REGION_NOT_LIVE_REGION",
            "REG_REGION_NOT_WORK_REGION",
            "LIVE_REGION_NOT_WORK_REGION",
            "REG_CITY_NOT_LIVE_CITY",
            "REG_CITY_NOT_WORK_CITY",
            "LIVE_CITY_NOT_WORK_CITY",
            #"EXT_SOURCE_1",
            #"EXT_SOURCE_2",
            #"EXT_SOURCE_3",
            "OBS_30_CNT_SOCIAL_CIRCLE",
            "DEF_30_CNT_SOCIAL_CIRCLE",
            "OBS_60_CNT_SOCIAL_CIRCLE",
            "DEF_60_CNT_SOCIAL_CIRCLE",
            "DAYS_LAST_PHONE_CHANGE",
            "FLAG_DOCUMENT_2",
            "FLAG_DOCUMENT_3",
            "FLAG_DOCUMENT_4",
            "FLAG_DOCUMENT_5",
            "FLAG_DOCUMENT_6",
            "FLAG_DOCUMENT_7",
            "FLAG_DOCUMENT_8",
            "FLAG_DOCUMENT_9",
            "FLAG_DOCUMENT_10",
            "FLAG_DOCUMENT_11",
            "FLAG_DOCUMENT_12",
            "FLAG_DOCUMENT_13",
            "FLAG_DOCUMENT_14",
            "FLAG_DOCUMENT_15",
            "FLAG_DOCUMENT_16",
            "FLAG_DOCUMENT_17",
            "FLAG_DOCUMENT_18",
            "FLAG_DOCUMENT_19",
            "FLAG_DOCUMENT_20",
            "FLAG_DOCUMENT_21",
            "AMT_REQ_CREDIT_BUREAU_HOUR",
            "AMT_REQ_CREDIT_BUREAU_DAY",
            "AMT_REQ_CREDIT_BUREAU_WEEK",
            "AMT_REQ_CREDIT_BUREAU_MON",
            "AMT_REQ_CREDIT_BUREAU_QRT",
            "AMT_REQ_CREDIT_BUREAU_YEAR",
        ]
        self.path = "../data/homecredit.csv"


DATASETS_ = {
    "german": GermanCredit,
    "taiwan": Taiwan,
    "adult": Adult,
    "acsincome": ACSIncome,
    "homecredit": HomeCredit,
}
