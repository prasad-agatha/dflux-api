from rest_framework import serializers
from dflux.db.models import Query

from .base import BaseModelSerializer


class QuerySerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the Query model data.
    """

    project = serializers.StringRelatedField()
    connection_name = serializers.ReadOnlyField(source="connection.name")
    connection_type = serializers.ReadOnlyField(source="connection.connection_type")
    excel_name = serializers.ReadOnlyField(source="excel.tablename")

    class Meta:
        model = Query
        fields = [
            "id",
            "project",
            "connection",
            "engine_type",
            "excel",
            "json",
            "excel_name",
            "connection_name",
            "connection_type",
            "name",
            # "extra",
            "created",
            "updated",
            "created_by",
        ]


class QueryDetailsSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the Query model data.
    """

    excel_name = serializers.ReadOnlyField(source="excel.tablename")

    class Meta:
        model = Query
        fields = "__all__"


class QueryPostSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the Query model data.
    """

    class Meta:
        model = Query
        fields = "__all__"


class NewQuerySerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the Query model data.
    """

    class Meta:
        model = Query
        fields = [
            "id",
            "connection",
            "engine_type",
            "excel",
            "name",
            "created",
            "updated",
        ]
