from rest_framework import serializers

from dflux.db.models import DataModel, DataModelMetaData

from .base import BaseModelSerializer


class DataModelSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the DataModel model data.
    """

    created_by = serializers.ReadOnlyField(source="user.email")

    class Meta:
        model = DataModel
        fields = [
            "id",
            "project",
            "user",
            "name",
            "model_type",
            "data",
            "other_params",
            "extra",
            "created",
            "created_by",
            "updated",
            "meta_data",
            "pickle_url",
            "scaler_url",
        ]


class RequiredFieldsDataModelSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the DataModel model data.
    """

    created_by = serializers.ReadOnlyField(source="user.email")

    class Meta:
        model = DataModel
        fields = [
            "id",
            "project",
            "user",
            "name",
            "model_type",
            "data",
            "other_params",
            "extra",
            "created",
            "created_by",
            "updated",
            "meta_data",
            "pickle_url",
            "scaler_url",
        ]


class DataModelLimitedFieldsSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the DataModel model data.
    """

    class Meta:
        model = DataModel
        fields = ["id", "project", "name", "model_type", "created"]


class DataModelMetaDataSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the DataModelMetaData model data.
    """

    class Meta:
        model = DataModelMetaData
        fields = "__all__"


class NewDataModelSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the DataModel model data.
    """

    class Meta:
        model = DataModel
        fields = [
            "name",
            "created",
        ]
