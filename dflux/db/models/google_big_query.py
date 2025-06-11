from django.db import models
from dflux.db.mixins import TimeAuditModel
from django.contrib.auth.models import User
from dflux.db.models import Connection


class GoogleBigQueryCredential(TimeAuditModel):
    """
    This model will allows store all the GoogleBigQueryCredential data in GoogleBigQueryCredential table.

    * This model contains FK(one to many) relation with Connection model.
    """

    connection = models.ForeignKey(
        Connection, on_delete=models.CASCADE, null=True, blank=True
    )
    credential_path = models.URLField()

    class Meta:
        verbose_name = "GoogleBigQueryCredential"
        verbose_name_plural = "GoogleBigQueryCredentials"
        db_table = "GoogleBigQueryCredentials"

    def __str__(self):
        return self.connection.name
