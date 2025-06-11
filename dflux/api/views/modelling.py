from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView

from django.shortcuts import Http404
from dflux.api import serializers
from dflux.api.serializers import DataModelMetaDataSerializer

import pandas as pd
from .auto_ml import (
    train_and_test_split,
    logistic_regression,
    support_vector_classifier,
    decision_tree,
    random_forest,
    xgboost,
    knn,
    naive_bayes_classifier,
    multinomailNB_classifier,
    adaboost_classifier,
    multi_layer_perceptron_classifier,
    model_evaluation_for_classification,
    linear_regression,
    support_vector_machine_regressor,
    decision_tree_regressor,
    random_forest_regressor,
    xgb_regressor,
    kneighbors_regressor,
    polynomial_regression,
    lasso_regressor,
    ridge_regressor,
    elasticnet_regression,
    sgd_regression,
    gradientboosting_regression,
    # lgbm_regression,
    # catboost_regression,
)


class TrainAndTestSplit(APIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert TrainAndTestSplit df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            # converting input data into df
            df = pd.DataFrame(data)
            X_train, X_test, y_train, y_test = train_and_test_split(df, target_variable)
            return Response(
                {
                    "x_train": X_train,
                    "x_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test,
                }
            )
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class LogisticRegressionModel(APIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert LogisticRegressionModel df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"logistic regression model": logistic_regression}
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            output_file_name = request.data.get("output_file_name")
            fit_model = method.get(request.data.get("fit_model"))

            # converting input data into df
            df = pd.DataFrame(data)
            (
                confusion_matrix_result,
                classification_report_result_mean,
                classification_report_result_reset_index,
                accuracy_score,
                labels_order,
                false_positive_rate,
                true_positive_rate,
                thresholds,
                auc_score,
                datetime,
                model_status,
            ) = model_evaluation_for_classification(
                df, target_variable, output_file_name, fit_model
            )
            return Response(
                {
                    "confusion_matrix_result": confusion_matrix_result,
                    "classification_report_result_mean": classification_report_result_mean,
                    "classification_report_result_reset_index": classification_report_result_reset_index,
                    "accuracy_score": accuracy_score,
                    "labels_order": labels_order,
                    "datetime": datetime,
                    "false_positive_rate": false_positive_rate,
                    "true_positive_rate": true_positive_rate,
                    "thresholds": thresholds,
                    "auc_score": auc_score,
                    "model_status": model_status,
                }
            )

        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class SupportVectorClassifier(APIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert SupportVectorClassifier df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"support vector classifier": support_vector_classifier}
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            output_file_name = request.data.get("output_file_name")
            fit_model = method.get(request.data.get("fit_model"))

            # converting input data into df
            df = pd.DataFrame(data)
            (
                confusion_matrix_result,
                classification_report_result_mean,
                classification_report_result_reset_index,
                accuracy_score,
                labels_order,
                false_positive_rate,
                true_positive_rate,
                thresholds,
                auc_score,
                datetime,
                model_status,
            ) = model_evaluation_for_classification(
                df, target_variable, output_file_name, fit_model
            )
            return Response(
                {
                    "confusion_matrix_result": confusion_matrix_result,
                    "classification_report_result_mean": classification_report_result_mean,
                    "classification_report_result_reset_index": classification_report_result_reset_index,
                    "accuracy_score": accuracy_score,
                    "labels_order": labels_order,
                    "false_positive_rate": false_positive_rate,
                    "true_positive_rate": true_positive_rate,
                    "thresholds": thresholds,
                    "auc_score": auc_score,
                    "datetime": datetime,
                    "model_status": model_status,
                }
            )
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class DecisionTreeClassifier(APIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert DecisionTreeClassifier df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"decision tree classifier": decision_tree}
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            output_file_name = request.data.get("output_file_name")
            fit_model = method.get(request.data.get("fit_model"))
            # converting input data into df
            df = pd.DataFrame(data)
            (
                confusion_matrix_result,
                classification_report_result_mean,
                classification_report_result_reset_index,
                accuracy_score,
                labels_order,
                false_positive_rate,
                true_positive_rate,
                thresholds,
                auc_score,
                datetime,
                model_status,
            ) = model_evaluation_for_classification(
                df, target_variable, output_file_name, fit_model
            )
            return Response(
                {
                    "confusion_matrix_result": confusion_matrix_result,
                    "classification_report_result_mean": classification_report_result_mean,
                    "classification_report_result_reset_index": classification_report_result_reset_index,
                    "accuracy_score": accuracy_score,
                    "labels_order": labels_order,
                    "false_positive_rate": false_positive_rate,
                    "true_positive_rate": true_positive_rate,
                    "thresholds": thresholds,
                    "auc_score": auc_score,
                    "datetime": datetime,
                    "model_status": model_status,
                }
            )
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class RandomForestClassifier(APIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert RandomForestClassifier df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"random forest classifier": random_forest}
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            output_file_name = request.data.get("output_file_name")
            fit_model = method.get(request.data.get("fit_model"))
            # converting input data into df
            df = pd.DataFrame(data)
            (
                confusion_matrix_result,
                classification_report_result_mean,
                classification_report_result_reset_index,
                accuracy_score,
                labels_order,
                datetime,
                false_positive_rate,
                true_positive_rate,
                thresholds,
                auc_score,
                model_status,
            ) = model_evaluation_for_classification(
                df, target_variable, output_file_name, fit_model
            )
            return Response(
                {
                    "confusion_matrix_result": confusion_matrix_result,
                    "classification_report_result_mean": classification_report_result_mean,
                    "classification_report_result_reset_index": classification_report_result_reset_index,
                    "accuracy_score": accuracy_score,
                    "labels_order": labels_order,
                    "datetime": datetime,
                    "false_positive_rate": false_positive_rate,
                    "true_positive_rate": true_positive_rate,
                    "thresholds": thresholds,
                    "auc_score": auc_score,
                    "model_status": model_status,
                }
            )
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class XGBClassifier(APIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert XGBClassifier df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"xgboost classifier": xgboost}
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            output_file_name = request.data.get("output_file_name")
            fit_model = method.get(request.data.get("fit_model"))
            # converting input data into df
            df = pd.DataFrame(data)
            (
                confusion_matrix_result,
                classification_report_result_mean,
                classification_report_result_reset_index,
                accuracy_score,
                labels_order,
                datetime,
                false_positive_rate,
                true_positive_rate,
                thresholds,
                auc_score,
                model_status,
            ) = model_evaluation_for_classification(
                df, target_variable, output_file_name, fit_model
            )
            return Response(
                {
                    "confusion_matrix_result": confusion_matrix_result,
                    "classification_report_result_mean": classification_report_result_mean,
                    "classification_report_result_reset_index": classification_report_result_reset_index,
                    "accuracy_score": accuracy_score,
                    "labels_order": labels_order,
                    "datetime": datetime,
                    "false_positive_rate": false_positive_rate,
                    "true_positive_rate": true_positive_rate,
                    "thresholds": thresholds,
                    "auc_score": auc_score,
                    "model_status": model_status,
                }
            )
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class KNNClassifier(APIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert KNNClassifier df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"knn classifier": knn}
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            output_file_name = request.data.get("output_file_name")
            fit_model = method.get(request.data.get("fit_model"))
            # converting input data into df
            df = pd.DataFrame(data)
            (
                confusion_matrix_result,
                classification_report_result_mean,
                classification_report_result_reset_index,
                accuracy_score,
                labels_order,
                false_positive_rate,
                true_positive_rate,
                thresholds,
                auc_score,
                datetime,
                model_status,
            ) = model_evaluation_for_classification(
                df, target_variable, output_file_name, fit_model
            )
            return Response(
                {
                    "confusion_matrix_result": confusion_matrix_result,
                    "classification_report_result_mean": classification_report_result_mean,
                    "classification_report_result_reset_index": classification_report_result_reset_index,
                    "accuracy_score": accuracy_score,
                    "labels_order": labels_order,
                    "datetime": datetime,
                    "false_positive_rate": false_positive_rate,
                    "true_positive_rate": true_positive_rate,
                    "thresholds": thresholds,
                    "auc_score": auc_score,
                    "model_status": model_status,
                }
            )
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class NaiveBayesClassifier(APIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert NaiveBayesClassifier df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"naive bayes classifier": naive_bayes_classifier}
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            output_file_name = request.data.get("output_file_name")
            fit_model = method.get(request.data.get("fit_model"))
            # converting input data into df
            df = pd.DataFrame(data)
            (
                confusion_matrix_result,
                classification_report_result_mean,
                classification_report_result_reset_index,
                accuracy_score,
                labels_order,
                false_positive_rate,
                true_positive_rate,
                thresholds,
                auc_score,
                datetime,
                model_status,
            ) = model_evaluation_for_classification(
                df, target_variable, output_file_name, fit_model
            )
            return Response(
                {
                    "confusion_matrix_result": confusion_matrix_result,
                    "classification_report_result_mean": classification_report_result_mean,
                    "classification_report_result_reset_index": classification_report_result_reset_index,
                    "accuracy_score": accuracy_score,
                    "labels_order": labels_order,
                    "datetime": datetime,
                    "false_positive_rate": false_positive_rate,
                    "true_positive_rate": true_positive_rate,
                    "thresholds": thresholds,
                    "auc_score": auc_score,
                    "model_status": model_status,
                }
            )
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class MultinomailNBClassifier(APIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert MultinomailNBClassifier df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"multinomailNB classifier": multinomailNB_classifier}
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            output_file_name = request.data.get("output_file_name")
            fit_model = method.get(request.data.get("fit_model"))
            # converting input data into df
            df = pd.DataFrame(data)
            (
                confusion_matrix_result,
                classification_report_result_mean,
                classification_report_result_reset_index,
                accuracy_score,
                labels_order,
                false_positive_rate,
                true_positive_rate,
                thresholds,
                auc_score,
                datetime,
                model_status,
            ) = model_evaluation_for_classification(
                df, target_variable, output_file_name, fit_model
            )
            return Response(
                {
                    "confusion_matrix_result": confusion_matrix_result,
                    "classification_report_result_mean": classification_report_result_mean,
                    "classification_report_result_reset_index": classification_report_result_reset_index,
                    "accuracy_score": accuracy_score,
                    "labels_order": labels_order,
                    "datetime": datetime,
                    "false_positive_rate": false_positive_rate,
                    "true_positive_rate": true_positive_rate,
                    "thresholds": thresholds,
                    "auc_score": auc_score,
                    "model_status": model_status,
                }
            )
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class AdaBoostClassifier(APIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert AdaBoostClassifier df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {"adaboost classifier": adaboost_classifier}
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            output_file_name = request.data.get("output_file_name")
            fit_model = method.get(request.data.get("fit_model"))
            # converting input data into df
            df = pd.DataFrame(data)
            (
                confusion_matrix_result,
                classification_report_result_mean,
                classification_report_result_reset_index,
                accuracy_score,
                labels_order,
                false_positive_rate,
                true_positive_rate,
                thresholds,
                auc_score,
                datetime,
                model_status,
            ) = model_evaluation_for_classification(
                df, target_variable, output_file_name, fit_model
            )
            return Response(
                {
                    "confusion_matrix_result": confusion_matrix_result,
                    "classification_report_result_mean": classification_report_result_mean,
                    "classification_report_result_reset_index": classification_report_result_reset_index,
                    "accuracy_score": accuracy_score,
                    "labels_order": labels_order,
                    "datetime": datetime,
                    "false_positive_rate": false_positive_rate,
                    "true_positive_rate": true_positive_rate,
                    "thresholds": thresholds,
                    "auc_score": auc_score,
                    "model_status": model_status,
                }
            )
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class MultiLayerPerceptronClassifier(APIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert MultiLayerPerceptronClassifier df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        method = {
            "multi layer perceptron classifier": multi_layer_perceptron_classifier
        }
        try:
            data = request.data.get("data")
            target_variable = request.data.get("target_variable")
            output_file_name = request.data.get("output_file_name")
            fit_model = method.get(request.data.get("fit_model"))
            # converting input data into df
            df = pd.DataFrame(data)
            (
                confusion_matrix_result,
                classification_report_result_mean,
                classification_report_result_reset_index,
                accuracy_score,
                labels_order,
                false_positive_rate,
                true_positive_rate,
                thresholds,
                auc_score,
                datetime,
                model_status,
            ) = model_evaluation_for_classification(
                df, target_variable, output_file_name, fit_model
            )
            return Response(
                {
                    "confusion_matrix_result": confusion_matrix_result,
                    "classification_report_result_mean": classification_report_result_mean,
                    "classification_report_result_reset_index": classification_report_result_reset_index,
                    "accuracy_score": accuracy_score,
                    "labels_order": labels_order,
                    "datetime": datetime,
                    "false_positive_rate": false_positive_rate,
                    "true_positive_rate": true_positive_rate,
                    "thresholds": thresholds,
                    "auc_score": auc_score,
                    "model_status": model_status,
                }
            )
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class ModellingMethods(APIView):
    """
    API endpoint that allows get the the result form the auto_ml.py corresponding function.
    - auto_ml.py stores all the ds related utility functions
    - This will convert ModellingMethods df into Json.

    * Requires JWT authentication.
    * This endpoint will allows only POST method.
    """

    permission_classes = (IsAuthenticated,)

    def post(self, request):
        modelling_methods = {
            "train_and_test_split": train_and_test_split,
            "logistic regression model": logistic_regression,
            "support vector classifier": support_vector_classifier,
            "decision tree classifier": decision_tree,
            "random forest classifier": random_forest,
            "xgboost classifier": xgboost,
            "knn classifier": knn,
            "naive bayes classifier": naive_bayes_classifier,
            "multinomailNB classifier": multinomailNB_classifier,
            "adaboost classifier": adaboost_classifier,
            "multi layer perceptron classifier": multi_layer_perceptron_classifier,
            # regression methods
            "linear_regression": linear_regression,
            "support_vector_machine_regressor": support_vector_machine_regressor,
            "decision_tree_regressor": decision_tree_regressor,
            "random_forest_regressor": random_forest_regressor,
            "xgb_regressor": xgb_regressor,
            "kneighbors_regressor": kneighbors_regressor,
            "polynomial_regression": polynomial_regression,
            "lasso_regressor": lasso_regressor,
            "ridge_regressor": ridge_regressor,
            "elasticnet_regression": elasticnet_regression,
            "sgd_regression": sgd_regression,
            "gradientboosting_regression": gradientboosting_regression,
            # "lgbm_regression": lgbm_regression,
            # "catboost_regression": catboost_regression,
        }
        # save meta data
        meta_data = request.data.get("meta_data")
        serializer = DataModelMetaDataSerializer(data=meta_data)
        if serializer.is_valid(raise_exception=True):
            meta_data_obj = serializer.save()
        output = {}
        input_modelling_methods = request.data.get("modelling")
        modeling_type = request.data.get("modeling_type")
        target_variable = request.data.get("target_variable")
        data = request.data.get("data")
        for input_modelling_method in input_modelling_methods:
            df = pd.DataFrame(data)
            modelling = modelling_methods.get(input_modelling_method)
            if modeling_type == "classification":
                output_file_name = (
                    f"{meta_data_obj.id}_{input_modelling_method}".replace(" ", "_")
                )
                try:
                    (
                        confusion_matrix_result,
                        classification_report_result_mean,
                        classification_report_result_reset_index,
                        accuracy_score,
                        labels_order,
                        false_positive_rate,
                        true_positive_rate,
                        thresholds,
                        auc_score,
                        X_test,
                        datetime,
                        model_status,
                        pickle_url,
                    ) = model_evaluation_for_classification(
                        df, target_variable, output_file_name, modelling
                    )
                    output[input_modelling_method] = {
                        "input_modelling_method": input_modelling_method,
                        "confusion_matrix_result": confusion_matrix_result,
                        "classification_report_result_mean": classification_report_result_mean,
                        "classification_report_result_reset_index": classification_report_result_reset_index,
                        "accuracy_score": accuracy_score,
                        "labels_order": labels_order,
                        "false_positive_rate": false_positive_rate,
                        "true_positive_rate": true_positive_rate,
                        "thresholds": thresholds,
                        "auc_score": [
                            {int(key): val} for key, val in auc_score.items()
                        ],
                        "X_test": X_test,
                        "datetime": datetime,
                        "model_status": model_status,
                        "pickle_url": pickle_url,
                    }
                except Exception as e:
                    output[input_modelling_method] = {
                        "input_modelling_method": input_modelling_method,
                        "confusion_matrix_result": [],
                        "classification_report_result_mean": {
                            "precision": 0,
                            "recall": 0,
                            "f1-score": 0,
                            "support": 0,
                        },
                        "classification_report_result_reset_index": [],
                        "accuracy_score": 0,
                        "error_msg": str(e),
                        "model_status": "Failed",
                    }
            elif modeling_type == "regression":
                try:
                    output_file_name = (
                        f"{meta_data_obj.id}_{input_modelling_method}".replace(" ", "_")
                    )
                    rmse_score, x_test, pickle_url = modelling(
                        df, target_variable, output_file_name
                    )
                    output[input_modelling_method] = {
                        "input_modelling_method": input_modelling_method,
                        "rmse_score": rmse_score,
                        "x_test": x_test,
                        "pickle_url": pickle_url,
                    }
                    # output[input_modelling_method].update("meta_data", serializer.data)
                except Exception as e:
                    return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        return Response({"output": output, "meta_data": serializer.data})
