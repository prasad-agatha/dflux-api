# Generated by Django 3.1.5 on 2021-05-14 05:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('db', '0050_auto_20210514_1124'),
    ]

    operations = [
        migrations.AlterField(
            model_name='charts',
            name='save_from',
            field=models.CharField(blank=True, choices=[('query', 'query'), ('data_model', 'data_model')], max_length=256, null=True),
        ),
    ]
