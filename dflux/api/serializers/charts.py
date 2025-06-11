from rest_framework import serializers

from dflux.db.models import Charts, Query, ShareCharts

from .base import BaseModelSerializer


class QueryAndConnectionSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the Query model data.
    """

    class Meta:
        model = Query
        fields = ["id", "connection", "name", "created", "updated", "raw_sql", "extra"]


class ChartSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the Charts model data.
    """

    created_by = serializers.ReadOnlyField(source="user.email")
    query = QueryAndConnectionSerializer()

    class Meta:
        model = Charts
        fields = [
            "id",
            "project",
            "user",
            "query",
            "data_model",
            "name",
            "chart_type",
            "save_from",
            "data",
            "extra",
            "created",
            "updated",
            "created_by",
        ]


class LimitedFieldsQueryAndConnectionSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the Query model data.
    """

    class Meta:
        model = Query
        fields = ["id"]


class LimitedFieldsChartSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the Charts model data.
    """

    query = LimitedFieldsQueryAndConnectionSerializer()
    created_by = serializers.ReadOnlyField(source="user.email")

    class Meta:
        model = Charts
        fields = [
            "id",
            "project",
            "user",
            "query",
            "data_model",
            "name",
            "chart_type",
            "save_from",
            # "data",
            "extra",
            "created",
            "updated",
            "created_by",
        ]


class ChartLimitedFieldsSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the Charts model data.
    """

    class Meta:
        model = Charts
        fields = ["id", "project", "user", "data_model", "name", "created", "updated"]


class SaveChartSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the Charts model data.
    """

    class Meta:
        model = Charts
        fields = [
            "id",
            "project",
            "user",
            "query",
            "data_model",
            "name",
            "chart_type",
            "save_from",
            "data",
            "extra",
            "created",
            "updated",
        ]


class ShareChartSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the ShareCharts model data.
    """

    class Meta:
        model = ShareCharts
        fields = "__all__"


class WantedFieldsChartSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the Charts model data.
    """

    class Meta:
        model = Charts
        fields = [
            "id",
        ]
