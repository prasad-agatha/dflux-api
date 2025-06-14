import json
import requests

from rest_framework import status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from django.db.models import Q
from django.shortcuts import get_object_or_404

from dflux.api.views.base import BaseAPIView
from dflux.api.serializers import JsonDataSerializer
from dflux.db.models import Connection, Project, JsonData

from .filters import JsonDataFilter
from .utils import create_table, insert_data_into_table, delete_table
from .permissions import (
    ProjectOwnerOrCollaborator,
    ProjectJsonDataAccess,
    ProjectModuleAccess,
)


class DumpJsonDataEndpoint(BaseAPIView):
    """
    API endpoint that allows dump Json files into the database.
    This endpoint will return list of all the Json tables.

    * Requires JWT authentication.
    * Only collaborator or owner of the project can access.
    * This endpoint will allows only GET, POST methods.
    """

    permission_classes = (IsAuthenticated, ProjectModuleAccess)

    def get(self, request, pk):
        """
        This method will allows return all the Json data related tables into the database.
        """
        try:
            queryset = JsonDataFilter(
                request.GET,
                queryset=JsonData.objects.select_related(
                    "user", "project", "connection"
                )
                .filter(
                    Q(user__id=request.user.id, project__id=pk)
                    | Q(project__id=pk, project__projectmembers__user=request.user),
                )
                .distinct(),
            ).qs
            serializer = JsonDataSerializer(queryset, many=True)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"msg": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    def post(self, request, pk):
        """
        This method will allows dump the Json file into the database.
        """
        try:
            project = Project.objects.get(id=pk)
            table_name = request.data.get("table_name").replace(" ", "_").lower()
            internal, is_created = Connection.objects.get_or_create(
                project=project,
                connection_type="internal",
            )
            json_file_path = request.data.get("json_file")
            response = requests.get(json_file_path)
            data = json.loads(response.content)

            columns = data[0].keys()
            records = [
                tuple([record.get(column) for column in columns]) for record in data
            ]
            create_table(table_name=table_name, columns=columns, file_type="json")
            JsonData.objects.get_or_create(
                user=request.user,
                project=project,
                connection=internal,
                tablename=table_name,
            )
            # dump data
            response = insert_data_into_table(
                table_name=table_name,
                columns=columns,
                records=records,
                file_type="json",
            )
            return Response({"msg": response})
        except Exception as e:
            delete_table(table_name)
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class DumpJsonDataDetail(BaseAPIView):
    """
    API endpoint that allows view, delete individual Json table data.

    * Requires JWT authentication.
    * Only collaborator or owner of the project can access.
    * This endpoint will allows only GET, DELETE methods.
    """

    permission_classes = (IsAuthenticated, ProjectJsonDataAccess)

    def delete(self, request, pk, format=None):
        """
        Delete Json table data.
        """
        json_data = get_object_or_404(JsonData, id=pk)
        delete_table(json_data.tablename)
        json_data.delete()
        return Response(
            {"message": "Delete Success"}, status=status.HTTP_204_NO_CONTENT
        )
