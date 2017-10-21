using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace CNTK
{
    public class CNTKExtension
    {
        public static Function Dense(Variable operand, int outputDim, DeviceDescriptor device, string outputName = "")
        {
            //flatten input layer if necessary
            if (operand.Shape.Rank != 1)
            {
                var newDim = operand.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                operand = CNTKLib.Reshape(operand, new int[] { newDim });
            }

            return FullyConnectedLinearLayer(operand, outputDim, device, outputName);
        }

        public static Function FullyConnectedLinearLayer(Variable input, int outputDim, DeviceDescriptor device,
            string outputName = "")
        {
            System.Diagnostics.Debug.Assert(input.Shape.Rank == 1);
            var inputDim = input.Shape[0];

            //dimensions of the weight matrix
            int[] s = { outputDim, inputDim };

            //create the weight Matrix
            var timesParam = new Parameter((NDShape)s, DataType.Float,
                CNTKLib.GlorotUniformInitializer(
                    CNTKLib.DefaultParamInitScale,
                    CNTKLib.SentinelValueForInferParamInitRank,
                    CNTKLib.SentinelValueForInferParamInitRank, 1),
                device, "timesParam");
            var timesFunction = CNTKLib.Times(timesParam, input, "times");

            //dimension of the bias weights
            int[] s2 = { outputDim };

            //create biases
            var plusParam = new Parameter(s2, 0.0f, device, "plusParam");
            return CNTKLib.Plus(plusParam, timesFunction, outputName);
        }
    }
}
