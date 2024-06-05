import numpy as np
import matplotlib.pyplot as plt


data = np.genfromtxt(
    "Admission_Predict_Ver1.1.csv", delimiter=",", names=True, dtype=None
)


def main():
    # 'data' is now a structured array with named columns

    field = data.dtype.names

    # seggregate each columns
    sno = data[field[0]]
    gre_score = data[field[1]]
    toefl_score = data[field[2]]
    university_rating = data[field[3]]
    sop = data[field[4]]
    lor = data[field[5]]
    cgpa = data[field[6]]
    research = data[field[7]]
    admission_chance = data[field[8]]

    # filtering the data for top universities
    rating_5_data = data[data["University_Rating"] == 5]
    filtered_field = (
        "Serial_No",
        "GRE_Score",
        "TOEFL_Score",
        "SOP",
        "LOR",
        "CGPA",
        "Research",
        "Chance_of_Admit",
    )

    filtered_variables = [
        rating_5_data["GRE_Score"],
        rating_5_data["TOEFL_Score"],
        rating_5_data["SOP"],
        rating_5_data["LOR"],
        rating_5_data["CGPA"],
        rating_5_data["Research"],
    ]
    filtered_chance = rating_5_data["Chance_of_Admit"]

    # forming M matrix, assuming admission chance also depends on a conatant value
    M = np.column_stack(
        [
            gre_score,
            toefl_score,
            university_rating,
            sop,
            lor,
            cgpa,
            research,
            np.ones(len(gre_score)),
        ]
    )
    coeffs = np.linalg.lstsq(M, admission_chance, rcond=False)[0]
    pred_chance = M @ coeffs
    variables = [gre_score, toefl_score, university_rating, sop, lor, cgpa, research]

    r_squared = r2(admission_chance, pred_chance)

    # correlation of params for all universities
    correlation_coeffs, top_variables = findCorrelation(
        variables, admission_chance, field
    )

    # correlation of params for top universities
    correlation_coeffs_top, top_params_rating5 = findCorrelation(
        filtered_variables, filtered_chance, filtered_field
    )
    print(f"R-squared Value = {r_squared}")
    print(
        f"Top 3 Parameters with more value for admission(all university) along with correlation value:\n{top_variables}"
    )
    print(
        f"Top 3 Parameters with more value for admission(top universities) along with correlation value:\n{top_params_rating5}"
    )

    plt.figure()
    plt.bar(
        [
            "GRE",
            "TOEFL",
            "UNVERSITY RATING",
            "SOP",
            "LOR",
            "CGPA",
            "RESEARCH",
        ],
        correlation_coeffs,
    )
    plt.title("Comparision of parameters using correlation")
    plt.xticks(rotation=27)
    plt.xticks(fontsize=6)
    plt.ylabel("correlation")
    values = [round(val, 3) for val in correlation_coeffs]
    # diaplaying values on bar graph
    for i, v in enumerate(values):
        plt.text(i, v + 0.001, str(v), ha="center", va="bottom", fontsize=10)
    plt.savefig("comparision_of_params.png")

    plt.figure()
    plt.bar(
        [
            "GRE",
            "TOEFL",
            "SOP",
            "LOR",
            "CGPA",
            "RESEARCH",
        ],
        correlation_coeffs_top,
    )
    plt.title("Comparision of parameters using correlation(for top universities)")
    plt.xticks(rotation=27)
    plt.xticks(fontsize=6)
    plt.ylabel("correlation")
    values = [round(val, 3) for val in correlation_coeffs_top]
    # displaying the values on bar graph
    for i, v in enumerate(values):
        plt.text(i, v + 0.001, str(v), ha="center", va="bottom", fontsize=10)
    plt.savefig("top_comparision_of_params.png")

    plt.figure()
    plt.scatter(
        admission_chance, pred_chance, color="red", label="Plot with all the data"
    )
    plt.plot(
        np.array([np.min(admission_chance), np.max(admission_chance)]),
        np.array([np.min(admission_chance), np.max(admission_chance)]),
        linestyle="--",
        label="y = x",
    )
    plt.xlabel("Given Admission Chance")
    plt.ylabel("Predicted Admission Chance")
    plt.legend()
    plt.savefig("admission_chance.png")


def findCorrelation(variables, admission_chance, field):
    correlation_coeffs = [np.corrcoef(var, admission_chance)[0, 1] for var in variables]

    # Find the top 3 variables with high correlation coeff.
    top_variables_indices = np.argsort(np.abs(correlation_coeffs))[::-1][:3]

    top_variables = [
        (field[i + 1], correlation_coeffs[i]) for i in top_variables_indices
    ]  # +1 to skip sno

    return (correlation_coeffs, top_variables)


def r2(y_true, y_pred):
    mean_y_true = np.mean(y_true)

    # Calculate Total Sum of Squares (TSS)
    tss = np.sum((y_true - mean_y_true) ** 2)

    # Calculate Residual Sum of Squares (RSS)
    rss = np.sum((y_true - y_pred) ** 2)

    # Calculate R-squared
    r_squared = 1 - (rss / tss)
    return r_squared


main()
