# Generated by Django 3.1.5 on 2021-02-10 07:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('db', '0011_auto_20210210_1155'),
    ]

    operations = [
        migrations.AlterField(
            model_name='query',
            name='vars',
            field=models.JSONField(blank=True, null=True),
        ),
    ]
