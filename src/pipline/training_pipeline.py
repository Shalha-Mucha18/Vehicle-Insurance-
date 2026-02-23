import sys
from src.exception import MyException
from src.logger import logging

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataValidationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact
from src.components.data_transformation import DataTransformation




class TrainPipeline:
    
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()


    def start_data_ingestion(self) -> DataIngestionArtifact:

        """
        This method of TrainPipeline class is responsible for starting data ingestion component
        """

        try:
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            logging.info("Getting the data from mongodb")

            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info("Got the train_set and test_set from mongodb")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")

            return data_ingestion_artifact

        except Exception as e:
            raise MyException(e, sys) from e    

    def start_data_validation(self,data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:


        logging.info("Entered the start_data_validation method of TrainPipeline class")

        try:
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                             data_validation_config=self.data_validation_config
                                             )

            data_validation_artifact = data_validation.initiate_data_validation()

            return data_validation_artifact

        except Exception as e:
            raise MyException(e, sys) from e  
    def start_data_transformation(self,data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:

        """
        This method of Trainpipeline class is responsible for starting data transformers
             
        """              
        try:
            logging.info("Entered the start_data_transformation method of TrainPipeline class")

            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,
                                                     data_transformation_config=self.data_transformation_config,
                                                     data_validation_artifact=data_validation_artifact)
            
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            
            return data_transformation_artifact

        except Exception as e:
            raise MyException(e, sys) from e
            



    def run_pipeline(self):
        try:
            logging.info("Entered the run_pipeline method of TrainPipeline class")
            logging.info("Starting the data ingestion component")
            data_ingestion_artifact = self.start_data_ingestion()
            logging.info("Data ingestion component completed")
            logging.info("Starting the data validation component")
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            logging.info("Data validation component completed")
            logging.info("Starting the data transformation component")
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,
                                                                         data_validation_artifact=data_validation_artifact)
            logging.info("Data transformation component completed")
            logging.info("Exited the run_pipeline method of TrainPipeline class")
        except Exception as e:
            raise MyException(e, sys) from e        
            
            

            

