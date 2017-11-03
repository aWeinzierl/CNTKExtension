using System.Linq;


namespace CNTK
{
    public static class CNTKExtension
    {
        public static Function Dense(Variable operand, int outputDim, CNTKDictionary weightInitializer, DeviceDescriptor device)
        {
            if (operand.Shape.Rank == 1)
                return FullyConnectedLinearLayer(operand, outputDim, device, weightInitializer);

            //flatten input layer
            var newDim = operand.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
            operand = CNTKLib.Reshape(operand, new[] { newDim });

            return FullyConnectedLinearLayer(operand, outputDim, device, weightInitializer);
        }

        public static Function Dense(Variable operand, int outputDim, CNTKDictionary weightInitializer, DeviceDescriptor device, string name)
        {
            //flatten input layer
            if (operand.Shape.Rank == 1)
                return FullyConnectedLinearLayer(operand, outputDim, device, weightInitializer, name);

            var newDim = operand.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
            operand = CNTKLib.Reshape(operand, new[] { newDim });

            return FullyConnectedLinearLayer(operand, outputDim, device, weightInitializer, name);
        }

        private static Function FullyConnectedLinearLayer(Variable input, int outputDim, DeviceDescriptor device,
            CNTKDictionary weightIntializer, string name = "")
        {
            System.Diagnostics.Debug.Assert(input.Shape.Rank == 1);

            var inputDim = input.Shape[0];

            int[] weightMatrixDimensions = { outputDim, inputDim };
            var weights = new Parameter(
                weightMatrixDimensions,
                DataType.Float,
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
