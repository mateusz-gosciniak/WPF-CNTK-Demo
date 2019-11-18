using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using CNTK;
using OxyPlot;
using OxyPlot.Series;

namespace WPF_CNTK_ws
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();

            var Data1 = new List<DataPoint>
            {
                new DataPoint(1, 3),
                new DataPoint(2, 5),
                new DataPoint(3, 7),
                new DataPoint(4, 9),
                new DataPoint(5, 11)
            };

            var ls1 = new LineSeries
            {
                MarkerType = MarkerType.Cross,
                MarkerSize = 5,
                MarkerStroke = OxyColors.Black,
                LineStyle = LineStyle.None
            };
            ls1.Points.Clear();
            ls1.Points.InsertRange(0, Data1);

            MyModel = new PlotModel { Title = "Example 1" };
            MyModel.Series.Add(ls1);
            oxyPlotView.Model = MyModel;


            var device = DeviceDescriptor.UseDefaultDevice();
            Output.Text += "Hello CNTK :) This is for " + device.Type + " only !!" + Environment.NewLine + device.Id + " is my device id" + Environment.NewLine;
        }

        public PlotModel MyModel { get; private set; }

        private static Trainer CreateTrainer(Function network, Variable target)
        {
            //learning rate
            var lrate = 0.082;
            var lr = new TrainingParameterScheduleDouble(lrate);
            //network parameters
            var zParams = new ParameterVector(network.Parameters().ToList());
            //create loss and eval
            Function loss = CNTKLib.SquaredError(network, target);
            Function eval = CNTKLib.SquaredError(network, target);
            //learners
            var llr = new List<Learner>();
            var msgd = Learner.SGDLearner(network.Parameters(), lr);
            llr.Add(msgd);
            var trainer = Trainer.CreateTrainer(network, loss, eval, llr);
            return trainer;
        }

        private static Function CreateLRModel(Variable x, DeviceDescriptor device)
        {
            var initializer = CNTKLib.GlorotUniformInitializer(1.0, 1, 0, 1);
            var bias = new Parameter(new NDShape(1, 1), DataType.Float, initializer, device, "b"); ;
            var weights = new Parameter(new NDShape(2, 1), DataType.Float, initializer, device, "w");
            var Wx = CNTKLib.Times(weights, x, "wx");
            var layer = CNTKLib.Plus(bias, Wx, "wx_b");
            return layer;
        }

        private void btLearn_Click(object sender, RoutedEventArgs e)
        {
            //Step 1: Create some Demo helpers
            Output.Text += "Linear Regression with CNTK!" + Environment.NewLine;
            Output.Text += "#### Linear Regression with CNTK! ####" + Environment.NewLine;
            Output.Text += "" + Environment.NewLine;
            //define device
            var device = DeviceDescriptor.UseDefaultDevice();

            //Step 2: define values, and variables
            Variable x = Variable.InputVariable(new int[] { 1 }, DataType.Float, "input");
            Variable y = Variable.InputVariable(new int[] { 1 }, DataType.Float, "output");

            //Step 2: define training data set from table above
            var xValues = Value.CreateBatch(new NDShape(1, 1), new float[] { 1f, 2f, 3f, 4f, 5f }, device);
            var yValues = Value.CreateBatch(new NDShape(1, 1), new float[] { 3f, 5f, 7f, 9f, 11f }, device);

            //Step 3: create linear regression model
            var lr = CreateLRModel(x, device);
            //Network model contains only two parameters b and w, so we query 
            //the model in order to get parameter values
            var paramValues = lr.Inputs.Where(z => z.IsParameter).ToList();
            var totalParameters = paramValues.Sum(c => c.Shape.TotalSize);
            Output.Text += $"LRM has {totalParameters} params, {paramValues[0].Name} and {paramValues[1].Name}." + Environment.NewLine;

            //Step 4: create trainer
            var trainer = CreateTrainer(lr, y);

            float b = 0;
            float A = 0;

            //Step 5: training
            for (int i = 1; i <= 500; i++)
            {
                var d = new Dictionary<Variable, Value>
                {
                    { x, xValues },
                    { y, yValues }
                };


                trainer.TrainMinibatch(d, true, device);

                var loss = trainer.PreviousMinibatchLossAverage();
                var eval = trainer.PreviousMinibatchLossAverage();

                if (i % 20 == 0)
                    Output.Text += $"It={i}, Loss={loss}, Eval={eval}" + Environment.NewLine;

                //print weights
                var b0_name = paramValues[0].Name;
                var b1_name = paramValues[1].Name;
                var b0 = new Value(paramValues[0].GetValue()).GetDenseData<float>(paramValues[0]);
                var b1 = new Value(paramValues[1].GetValue()).GetDenseData<float>(paramValues[1]);
                if (i == 500)
                {
                    Output.Text += $" " + Environment.NewLine;
                    Output.Text += $"Training process finished with the following regression parameters:" + Environment.NewLine;
                    Output.Text += $"b={b0[0][0]}, w={b1[0][0]}" + Environment.NewLine;
                    Output.Text += $" " + Environment.NewLine;
                }

                b = b0[0][0];
                A = b1[0][0];
            }

            MyModel.Series.Add(new FunctionSeries(LinearFunction(A, b), -1, 12, 0.01, "result"));
            MyModel.InvalidatePlot(true);
        }

        private Func<double, double> LinearFunction(float A, float b) => x => A* x + b;
    }
}
