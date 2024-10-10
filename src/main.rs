use gnuplot::{
    AxesCommon, Figure,
    PlotOption::{Color, PointSize, PointSymbol},
};
use machine_learning::{util::csv, KMeanClustering};

fn main() {
    let xs = csv::load_data_only::<f64>("cluster_data.csv").unwrap();

    let model = KMeanClustering::train(xs.clone(), 4, 1000);
    let xs_class = model.predict(xs.clone());

    // Plot the points
    let class_colors = ["dark-red", "dark-blue", "dark-green", "dark-orange"];
    let class_shape = ['S', 'O', 'T', 'R'];
    let mut fg = Figure::new();
    let graph = fg.axes2d().set_title("Data", &[]);

    for (class, mean) in model.means.row_iter().enumerate() {
        graph.points(
            vec![mean[0]],
            vec![mean[1]],
            &[
                Color(class_colors[class]),
                PointSymbol(class_shape[class]),
                PointSize(3.),
            ],
        );
    }

    for (i, row) in xs.row_iter().enumerate() {
        graph.points(
            vec![row[0]],
            vec![row[1]],
            &[
                Color(class_colors[xs_class[i]]),
                PointSymbol(class_shape[xs_class[i]]),
            ],
        );
    }

    fg.show().unwrap();
}
