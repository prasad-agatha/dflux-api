# Generated by Django 3.1.5 on 2021-04-22 12:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('db', '0025_auto_20210420_0919'),
    ]

    operations = [
        migrations.AddField(
            model_name='query',
            name='description',
            field=models.TextField(blank=True, null=True),
        ),
    ]
