from sklearn import preprocessing

class CategoricalFeatures:
    def __init__(self, df, categorical_features, encoding_type, handle_na = False):
        """
        df: pandas dataframe
        categorical features: list of col names , e.g ["bin_1", "nom_1", "ord_1"]
        encoding_type: label, binary, ohe
        handle_na : True/False

        """
        self.df = df  
        self.categorical_features = categorical_features
        self.encoding_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None

        if self.handle_na:
            for c in self.categorical_features:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-99999999")
                # deep = True:  any change on the data on original will not be reflected. 
                #Take a note what you want to and change accordingly.
        self.output_df = self.df.copy(deep = True)

        #-------------------------------------
        #fit() vs transform()
        # fit encodes all the columns that you pass and when you pass transforms, it replaces the original column
        # with newly 

    def _label_encoding(self):
        for c in self.categorical_features:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:,c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df


    def _label_binarization(self):
        for c in self.categorical_features:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.df[c].values)
            val = lbl.transform(self.df[c].values)
            self.output_df = self.output_df.drop(c, axis=1) 
            for j in range(val.shape[1]):
                new_col_name = c+f"__bin{j}"
                self.output_df[new_col_name] = val[:, j] 
            self.binary_encoders[c] = lbl
        return self.output_df


    def _one_hot(self):
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.df[self.categorical_features].values)
        return ohe.transform(self.df[self.categorical_features].values)


    def fit_transform(self):
        if self.encoding_type=="label":
            return self._label_encoding()
        elif self.encoding_type=="binary":
            return self._label_binarization()
        elif self.encoding_type=="ohe":
            return self._one_hot()
        else:
            raise Exception ("Sorry did not understand the encoding type")
    
    
    def transform(self, dataframe):
        if self.handle_na:
            for c in self.categorical_features:
                dataframe.loc[:,c] = dataframe.loc[:,c].astype(str).fillna("-99999999")

                # TODO  add other impute strategies here.
        if self.encoding_type == "label":
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:,c] = lbl.transform(dataframe[c].values)
            return dataframe

        elif self.encoding_type == "binary":
            for c, lbl in self.binary_encoders.items():
                val = lbl.transform(dataframe[c].values)
                dataframe = dataframe.drop(c, axis =1)
                for j in range(val.shape[1]):
                    new_col_name = c + f"__bin__{j}"
                    dataframe[new_col_name] = val[:,j]
            return dataframe

        elif self.encoding_type == "ohe":
            return self.ohe(dataframe[self.categorical_features].values)

        else:
            raise Exception ("Sorry did not understand the encoding type")









            


