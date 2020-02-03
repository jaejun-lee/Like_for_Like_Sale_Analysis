import psycopg2
import pandas as pd
import pandas.io.sql as sqlio
from datetime import datetime

class Pipeline(object):
    """Impulsify Database ETL
    """

    SQL_STORE_SALES = '''SELECT
    id,
    store_id,
    property_id,
    property_store_id,
    tendered,
    sale_date
    FROM
        carts
    WHERE
        tendered IS NOT NULL AND
        sale_date BETWEEN '2017-01-01' AND '2019-12-31'
    '''
    SQL_STORE_PROPERTY = '''SELECT
    property_stores.id,
    stores.id as store_id, 
    stores.name as store_name, 
    stores.activation_date,
    props.id as property_id,
    props.property_code,
    props.address,
    props.city,
    props.state,
    props.zip,
    props.name,
    props.kind,
    props.time_zone,
    props.location_type,
    props.rooms,
    props.flag_name,
    props.brand_name
    FROM 
        property_stores
    LEFT JOIN 
        stores 
    ON property_stores.store_id = stores.id
    LEFT JOIN
        (SELECT
        properties.id,
        properties.property_code,
        properties.address,
        properties.city,
        properties.state,
        properties.zip,
        properties.name,
        properties.kind,
        properties.time_zone,
        properties.location_type,
        properties.rooms,
        flags.id as flag_id,
        flags.name as flag_name,
        brands.id as brand_id,
        brands.name as brand_name
        FROM properties
        join flags ON properties.flag_id = flags.id
        join brands ON properties.brand_id = brands.id
        WHERE
        properties.currency_id = 1) AS props
    ON property_stores.property_id = props.id
    '''

    def __init__(self, conn, ):
        """
        Parameters
        ----------
        conn : SQL connection object
        Returns
        -------
        None
        """
        self.conn = conn
        self.c = conn.cursor()

    def load_store_sales(self):
        """load store_sales to dataframe
        Parameters
        ----------
        Returns
        -------
        None
        """
        self.store_sales = sqlio.read_sql_query(Pipeline.SQL_STORE_SALES, self.conn)

    
    def load_store_property(self):
        """load store_property to dataframe
        Parameters
        ----------
        Returns
        -------
        None
        """
        self.store_property = sqlio.read_sql_query(Pipeline.SQL_STORE_PROPERTY, self.conn)

    def close(self):
        self.conn.close()

    def save_store_sales(self, file_path="../data/store_sales.pkl"):
        self.store_sales.to_pickle(path=file_path)

    def save_store_property(self, file_path="../data/store_property.pkl"):    
        self.store_property.to_pickle(path=file_path)
    
    def save_data(self, file_path="../data/data.pkl"):
        self.data.to_pickle(path=file_path)


    def load_data(self):
        """JOIN store_sales and store_properties to build dataset
        Parameters
        ----------
        Returns
        -------
        None
        """
        self.store_sales['month'] = self.store_sales.sale_date.dt.month
        self.store_sales['year'] = self.store_sales.sale_date.dt.year
        self.store_sale_by_month = self.store_sales.groupby(by=["store_id", "year", "month"]).agg({'tendered':np.sum}).reset_index()
        self.store_sale_average = self.store_sale_by_month.groupby(by="store_id").agg({"tendered":np.mean}).reset_index()
        self.store_property = self.store_property.dropna()
        #self.store_property = self.store_property.fillna({"store_id":0, "property_id":0, "rooms":0})
        self.store_property = self.store_property.astype({"store_id":'int64', "property_id":"int64", "rooms":"int64"})
        self.data = self.store_sale_average.merge(self.store_property, on="store_id", how='left')
        self.data = self.data.dropna()
        self.data = self.data.astype({"property_id":"int64", "rooms":"int64"})
        self.data.pop("id")

        #pipeline.data.groupby(by="flag_name").store_id.count().plot()


if __name__ == '__main__':

    conn = psycopg2.connect(dbname='impulsify', user='postgres', 
                            host='localhost', port='5432')
    #today = '2014-08-14'
    #date = datetime.strptime(today, '%Y-%m-%d').strftime("%Y%m%d")

    pipeline = Pipeline(conn)
    pipeline.load_store_sales()
    pipeline.load_store_property()
    pipeline.load_data()
    pipeline.save_data()
    pipeline.save_store_sales()
    pipeline.save_store_property()
    pipeline.close()