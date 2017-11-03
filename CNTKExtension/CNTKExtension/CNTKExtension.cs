using System.Linq;


namespace CNTK
{
    public static class CNTKExtension
    {
        public static Function Dense(Variable operand, int outputDim, DataType dataType, CNTKDictionary weightInitializer, DeviceDescriptor device)
        {
            return Dense(operand, outputDim, dataType, weightInitializer, device, "");
        }

        public static Function Dense(Variable operand, int outputDim, DataType dataType, CNTKDictionary weightInitializer, DeviceDescriptor device, string name)
        {
            //flatten input layer
            if (operand.Shape.Rank == 1)
                return FullyConnectedLinearLayer(operand, outputDim, dataType, device, weightInitializer, name);

            var newDim = operand.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
            operand = CNTKLib.Reshape(operand, new[] { newDim });

            return FullyConnectedLinearLayer(operand, outputDim, dataType, device, weightInitializer, name);
        }

        private static Function FullyConnectedLinearLayer(Variable input, int outputDim, DataType dataType, DeviceDescriptor device,
            CNTKDictionary weightIntializer, string name = "")
        {
            System.Diagnostics.Debug.Assert(input.Shape.Rank == 1);

            var inputDim = input.Shape[0];

            int[] weightMatrixDimensions = { outputDim, inputDim };
            var weights = new Parameter(
                weightMatrixDimensions,
                dataType,
                weightIntializer,
                device,
                "weights");
            var timesFunction = CNTKLib.Times(weights, input, "times");

            int[] biasDimension = { outputDim };
            var bias = new Parameter(biasDimension, 0.0f, device, "plusParam");

            return CNTKLib.Plus(bias, timesFunction, name);
        }

        public static bool ReachedEndOfEpoch(this UnorderedMapStreamInformationMinibatchData minibatchData)
        {
            return minibatchData.Values.Any(stream => stream.sweepEnd);
        }
    }
}
