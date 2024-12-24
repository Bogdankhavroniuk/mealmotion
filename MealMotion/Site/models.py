

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np
import pulp as pl
import json

class Calories():

    def __init__(self , weight  , sex  , age , height ,  activity ,  time_in_hours_activity, time_from_last_eating , file_path = "Site/Sourcess/exercise_dataset.csv"):

        self.weight =  weight
        self.activity = activity
        self.time_in_hours_activity= time_in_hours_activity
        self.time_from_last_eating = time_from_last_eating
        self.sex = sex
        self.age = age
        self.height = height
        self.calories = None
        self.fle_path = file_path

    def predict_calories(self ):

        file_path =  self.fle_path

        # Load the CSV into a DataFrame
        df = pd.read_csv(file_path)

        # Rename the 'Activity, Exercise or Sport (1 hour)' column to 'Activity'
        df.rename(columns={'Activity, Exercise or Sport (1 hour)': 'Activity'}, inplace=True)



        # Reshaping the dataframe (melting)
        df_melted = df.melt(id_vars=['Activity'],
                            value_vars=['130 lb', '155 lb', '180 lb', '205 lb'],
                            var_name='Weight',
                            value_name='Calories Burned')

        # Convert weight from pounds to kilograms
        df_melted['Weight'] = df_melted['Weight'].str.extract('(\d+)').astype(float) * 0.453592


        # Features and target
        X = df_melted[['Activity', 'Weight']]
        y = df_melted['Calories Burned']

        # Preprocessing: Handling categorical variables with OneHotEncoder and ignoring unknown categories
        preprocessor = ColumnTransformer(
            transformers=[
                ('activity', OneHotEncoder(handle_unknown='ignore'), ['Activity'])
            ], remainder='passthrough'
        )

        # Building the pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ])

        # Splitting data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        pipeline.fit(X_train, y_train)


        return pipeline.predict(pd.DataFrame({'Activity': [self.activity], 'Weight': [self.weight]}))[0]

    def get_calories_burn(self):


        if  self.sex == "Male" :
            BRM = 655.1 + (9.563 *  self.weight) + (1.85 * self.height ) - (4.676 * self.age )
        else :
            BRM = 66.47 + (13.75 *  self.weight) + (5.003 * self.height) - (6.755 * self.age)

        Totall_calories = lambda time_in_hours_activity , model1_results , time_from_last_eating, BRM : time_in_hours_activity * model1_results + time_from_last_eating/24 * BRM

        self.calories = Totall_calories(self.time_in_hours_activity  , self.predict_calories(), self.time_from_last_eating , BRM)
        return self.calories






class MealPlanner:
    def __init__(self, weight , calories_burned , goal):

        self.weight = weight
        self.calories_burned = calories_burned
        self.goal  =  goal


    def calculate_meal(self):
        """
        Calculate meal requirements based on user inputs.

        :return: Dictionary with meal details
        """
        # Base calorie range (customize logic as needed)
        if self.goal == "gain_weight" :
           cal_lo = self.calories_burned + 50
           cal_up = self.calories_burned + 100
        else  :
            cal_lo = self.calories_burned  - 100
            cal_up = self.calories_burned - 50

        # Protein range (approximation: 0.8g to 1g per kg of body weight)
        pro_lo = round(0.8 * self.weight, 1)
        pro_up = round(1.0 * self.weight, 1)

        # Fat range (approximation: 0.4g to 0.6g per kg of body weight)
        fat_lo = round(0.4 * self.weight, 1)
        fat_up = round(0.6 * self.weight, 1)

        # Sodium range (based on general dietary recommendations)
        sod_lo = 2300
        sod_up = 2500

        return {
            "cal_up": cal_up,
            "cal_lo": cal_lo,
            "pro_up": pro_up,
            "pro_lo": pro_lo,
            "fat_up": fat_up,
            "fat_lo": fat_lo,
            "sod_up": sod_up,
            "sod_lo": sod_lo,
        }


class Recomendationmodel():

    def __init__(self, file_path = "Site\Sourcess\epi_r.csv"):
        self.file_path = "Site\Sourcess\epi_r.csv"
        self.data = None
        self.text  = None
    def read_csv(self):
        """
        Зчитує CSV-файл та зберігає його вміст у self.data.
        """
        try:
            self.data = pd.read_csv(self.file_path)
            print("Дані успішно зчитано.")
        except Exception as e:
            print(f"Помилка при зчитуванні CSV: {e}")
            self.data = None



    def filter_data(self ):
        """
        Фільтрує страви з калорійністю менше max_calories та повертає список бібліотек,
        де кожен запис представлений як {назва_колонки: значення}.
        """
        self.data = self.data.drop_duplicates()
        self.results =  None
        self.text = None

        def cap_outliers(df_cleaned, column_name):
            Q1 = self.data[column_name].quantile(0.25)
            Q3 = self.data[column_name].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df_cleaned[column_name] = np.clip(df_cleaned[column_name], lower_bound, upper_bound)

        numerical_cols = ['rating', 'calories', 'protein', 'fat', 'sodium']
        for col in numerical_cols:
            cap_outliers(self.data, col)

        columns_to_impute = ['calories', 'protein', 'fat', 'sodium']
        knn_imputer = KNNImputer(n_neighbors=3)
        imputed_data = knn_imputer.fit_transform(self.data[columns_to_impute])
        self.data[columns_to_impute] = imputed_data


    def recipe_recommend(self, number_of_dishes, number_of_candidates, nut_conf):
        tmp = self.data.copy()
        candidates_list = []

        # create recommend-list {number_of_candidates} times
        for i in range(0, number_of_candidates):

            m = pl.LpProblem(sense=pl.LpMaximize)
            tmp.loc[:, 'v'] = [pl.LpVariable('x%d' % i, cat=pl.LpBinary) for i in range(len(tmp))]
            m += pl.lpDot(tmp["rating"], tmp["v"])
            m += pl.lpSum(tmp["v"]) <= number_of_dishes
            m += pl.lpDot(tmp["calories"], tmp["v"]) >= nut_conf["cal_lo"]
            m += pl.lpDot(tmp["calories"], tmp["v"]) <= nut_conf["cal_up"]
            m += pl.lpDot(tmp["protein"], tmp["v"]) >= nut_conf["pro_lo"]
            m += pl.lpDot(tmp["protein"], tmp["v"]) <= nut_conf["pro_up"]
            m += pl.lpDot(tmp["fat"], tmp["v"]) >= nut_conf["fat_lo"]
            m += pl.lpDot(tmp["fat"], tmp["v"]) <= nut_conf["fat_up"]
            m += pl.lpDot(tmp["sodium"], tmp["v"]) >= nut_conf["sod_lo"]
            m += pl.lpDot(tmp["sodium"], tmp["v"]) <= nut_conf["sod_up"]
            m.solve(pl.PULP_CBC_CMD(msg=0, options=['maxsol 1']))

            if m.status == 1:
                tmp.loc[:, 'val'] = tmp["v"].apply(lambda x: pl.value(x))
                ret = tmp.query('val==1')["title"].values
                candidates_list.append(ret)
                # update dataframe (remove recommended title )
                tmp = tmp.query('val==0')
        self.results = candidates_list

        return candidates_list

    def get_text(self):
        text = ""
        for ind, rec in enumerate(self.results):
            text += str(ind + 1) + ") "
            for ind ,  r in enumerate(rec):
                if ind <len(rec)-1:
                  text += str(r) + "\t+\t"
                else :
                   text += str(r)
            text += "\n"
        self.text = text

class Dish():


    def __init__(self , name  ):
        self.name  =  name
        self.recepie = None

    def get_directions_by_title(self):
        try:
            # Load the JSON file
            with open("project/Data_processing/Sourcess/full_format_recipes.json", 'r') as file:
                data = json.load(file)

            # Iterate through the root items
            for item in data.get('root', []):
                if item.get('title') == self.name:
                    self.recepie = item.get('directions', 'No directions found')
                    return item.get('directions', 'No directions found')


            return f"No item with the title '{self.name}' found."

        except FileNotFoundError:
            return "The specified JSON file was not found."
        except json.JSONDecodeError:
            return "Error decoding JSON. Please check the file format."


def test():
    calories_model = Calories(
        weight=70,
        sex="Male",
        age=30,
        height=175,
        activity="Running",
        time_in_hours_activity=1,
        time_from_last_eating=4,
        file_path="Sourcess/exercise_dataset.csv"
    )

    # Predict calories burned
    calories_burned = calories_model.get_calories_burn()
    print(f"Calories burned: {calories_burned:.2f}")

    # Initialize the MealPlanner
    meal_planner = MealPlanner(
        weight=wwight,
        calories_burned=calories_burned,
        goal="gain_weight"  # or "lose_weight"
    )

    # Calculate the meal plan
    meal_plan = meal_planner.calculate_meal()
    print("Meal Plan:", meal_plan)

    # Initialize the RecommendationModel
    recommendation_model = Recomendationmodel(file_path="Sourcess/epi_r.csv")

    # Read the recipe data
    recommendation_model.read_csv()


    # Filter the data
    recommendation_model.filter_data()

    # Generate recipe recommendations
    recommendation_model.recipe_recommend(
        number_of_dishes=3,
        number_of_candidates=10,
        nut_conf=meal_plan)

    recommendation_model.get_text()
    recomendation = recommendation_model.results
    text = recommendation_model.text

    for i in recomendation:
        for name in i :
            Dishn  =   Dish(name)
            recepie = Dishn.recepie
            print(recepie)

    print(text)