from rest_framework import serializers

from dflux.db.models import DashBoard, DashBoardCharts, ShareDashBoard
from .charts import ChartSerializer, WantedFieldsChartSerializer

from .base import BaseModelSerializer


class DashBoardChartsSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the DashBoardCharts model data.
    """

    chart = ChartSerializer()

    class Meta:
        model = DashBoardCharts
        fields = [
            "id",
            # "dashboard",
            "chart",
            "height",
            "width",
            "position_x",
            "position_y",
            "extra",
        ]


class DashBoardSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the DashBoard model data.
    """

    user = serializers.StringRelatedField()
    created_by = serializers.ReadOnlyField(source="user.email")
    charts = DashBoardChartsSerializer(many=True)

    class Meta:
        model = DashBoard
        fields = [
            "id",
            "name",
            "user",
            "created_by",
            "description",
            "created",
            "updated",
            "charts",
            "extra",
        ]


class RequiredFieldsDashBoardSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the DashBoard model data.
    """

    user = serializers.StringRelatedField()
    created_by = serializers.ReadOnlyField(source="user.email")
    charts = DashBoardChartsSerializer(many=True)

    class Meta:
        model = DashBoard
        fields = [
            "id",
            "name",
            "user",
            "created_by",
            "description",
            "created",
            "updated",
            "charts",
            "extra",
        ]


class RequiredFieldsDashBoardChartsSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the DashBoardCharts model data.
    """

    chart = WantedFieldsChartSerializer()

    class Meta:
        model = DashBoardCharts
        fields = [
            "id",
            # "dashboard",
            "chart",
            # "height",
            # "width",
            # "position_x",
            # "position_y",
            # "extra",
        ]


class DashBoardDetailSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the DashBoard model data.
    """

    charts = DashBoardChartsSerializer(many=True)

    class Meta:
        model = DashBoard
        fields = [
            "id",
            "name",
            "user",
            "description",
            "created",
            "updated",
            "charts",
            "extra",
        ]


class SaveDashBoardSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the DashBoard model data.
    """

    class Meta:
        model = DashBoard
        fields = ["name", "project", "user", "description", "extra"]


class SaveDashBoardChartsSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the DashBoardCharts model data.
    """

    class Meta:
        model = DashBoardCharts
        fields = [
            "dashboard",
            "chart",
            "height",
            "width",
            "position_x",
            "position_y",
            "extra",
        ]


class ShareDashBoardSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the ShareDashBoard model data.
    """

    class Meta:
        model = ShareDashBoard
        fields = "__all__"


class NewDashBoardSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the DashBoard model data.
    """

    class Meta:
        model = DashBoard
        fields = [
            "name",
            "created",
        ]
