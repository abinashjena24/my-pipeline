
import logging
from typing import Dict,List,Optional,Tuple,Self
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin


logger=logging.getLogger(__name__)
class IQROutlierCapper(BaseEstimator,TransformerMixin):
    def __init__(self,columns=None,factor=1.5)->None:
        self.columns: Optional[List[str]]=columns
        self.factor: float=factor
        self.caps_: Dict[str, Tuple[float, float]]={}
    def fit(self,x:pd.DataFrame,y:Optional[pd.Series]=None)-> Self:
        if not isinstance(x,pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        if self.columns is not None:
            target_cols=self.columns
        else:
            target_cols=x.select_dtypes(include=(np.number)).columns
        for col in target_cols:
                q1=x[col].quantile(0.25)
                q3=x[col].quantile(0.75)
                iqr=q3-q1
                lower_bound=float(q1-(self.factor*iqr))
                upper_bound=float(q3+(self.factor*iqr))
                self.caps_[col]=(lower_bound,upper_bound)
        return self
    def transform(self,x:pd.DataFrame)->pd.DataFrame:
        if not self.caps_:
            raise RuntimeError("Transformer has not been fitted yet. Call fit() first.")
        if not isinstance(x,pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")