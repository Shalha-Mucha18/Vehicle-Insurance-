from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.entity.estimator import MyModel
import sys
from pandas import DataFrame

class Proj1Estimator:

    """
    This class is used to save and retrieve the model from s3 and to do prediction on the model.

    """

    def __init__(self,bucket_name:str,model_path:str):

        """
        :param bucket_name: Name of the s3 bucket
        :param model_path: Path to the model in the s3 bucket
        """
        self.bucket_name = bucket_name
        self.s3 = SimpleStorageService()

        self.model_path = model_path
        self.loaded_model:MyModel = None

    def is_model_present(self,model_path:str) -> bool:
        try:

            return self.s3.s3_key_path_available(bucket_name=self.bucket_name,s3_key=model_path)
        except MyException as e:
            print(e)
            return False

    def load_model(self,model_path):
        try:
            return self.s3.load_model(model_name=model_path, bucket_name=self.bucket_name)
        except MyException as e:
            print(e)
            return None

    def save_model(self,from_file,remove:bool=False)-> None:
         """
        save the model to the model path
        :param from_file: Path to the model in the local system
        :param remove: Whether to remove the model from the local system after saving
        :return: None
         """   
         try:
            self.s3.upload_file(from_filename=from_file,
            to_filename=self.model_path,
            bucket_name=self.bucket_name,remove=remove)
         except Exception as e:
            raise MyException(e, sys) from e                   

    def predict(self,data_frame:DataFrame):
        """
        : param dataframe
        : return

        """
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model(self.model_path)
            return self.loaded_model.predict(data_frame)
        except Exception as e:
            raise MyException(e, sys) from e