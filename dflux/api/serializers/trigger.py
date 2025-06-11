from dflux.db.models import Trigger, TriggerOutput, ChartTrigger

from .base import BaseModelSerializer


class TriggrSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the Trigger model data.
    """

    class Meta:
        model = Trigger
        fields = "__all__"


class TriggerOutputSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the TriggerOutput model data.
    """

    class Meta:
        model = TriggerOutput
        fields = "__all__"


class ChartTriggrSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the ChartTrigger model data.
    """

    class Meta:
        model = ChartTrigger
        fields = "__all__"


class NewChartTriggrSerializer(BaseModelSerializer):
    """
    This serializer will allows serialize the ChartTrigger model data.
    """

    class Meta:
        model = ChartTrigger
        fields = [
            "name",
            "created",
        ]
