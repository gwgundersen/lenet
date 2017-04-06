-- ============================================================================
-- Train LeNet5 classifier on MNIST.
-- ============================================================================

require 'svhn'
require 'optim'
local lenet = require 'lenet'


-------------------------------------------------------------------------------
-- Parameters and constants.

local N_EPOCHS           = 10
local BATCH_SIZE         = 20
local N_TRAINING_BATCHES = 60000
local N_TESTING_BATCHES  = 10000
local MEAN               = 0
local STD                = 1
local CLASSES            = {'1','2','3','4','5','6','7','8','9','10'}

local model = lenet.model
local GEOMETRY = lenet.geometry
local criterion = nn.ClassNLLCriterion()  -- Negative log likelihood.

-- Create training set and normalize
local trainData = svhn.loadTrainSet()
local testData = svhn.loadTestSet()

local parameters, gradParameters = model:getParameters()
local confusion = optim.ConfusionMatrix(CLASSES)


-------------------------------------------------------------------------------
-- Batch stochastic gradient descent.

local function getInputsAndTargetsTensors(dataset, t)
    local inputs = torch.Tensor(
        BATCH_SIZE,
        GEOMETRY[1],
        GEOMETRY[2],
        GEOMETRY[3]
    )
    local targets = torch.Tensor(BATCH_SIZE)
    local k = 1
    for i = t, math.min(t+BATCH_SIZE-1, dataset:size()) do
        local sample = dataset[i]
        local input = sample[1]:clone()
        local _, target = sample[2]:clone():max(1)
        target = target:squeeze()
        inputs[k] = input
        targets[k] = target
        k = k + 1
    end
    return inputs, targets
end


local function train(dataset)
    print('Training...')
    for t = 1, dataset:size(), BATCH_SIZE do
        local inputs, targets = getInputsAndTargetsTensors(dataset, t)

        -- Optimization functions take closure with access to model and data.
        local function f_eval(x)
            -- Reset gradients.
            gradParameters:zero()
            local outputs = model:forward(inputs)
            local f       = criterion:forward(outputs, targets)
            local df_do   = criterion:backward(outputs, targets)
            model:backward(inputs, df_do)

            for i = 1, BATCH_SIZE do
                confusion:add(outputs[i], targets[i])
            end

            return f, gradParameters
        end

        optim.adam(f_eval, parameters)
    end
end


local function test(dataset)
    print('Testing...')
    for t = 1, dataset:size(), BATCH_SIZE do
        -- disp progress
        xlua.progress(t, dataset:size())
        local inputs, targets = getInputsAndTargetsTensors(dataset, t)
        local preds = model:forward(inputs)

        for i = 1, BATCH_SIZE do
            confusion:add(preds[i], targets[i])
        end
    end
    return confusion
end


for i = 1, N_EPOCHS do
    print('Epoch', i)
    train(trainData)
end

local confusion = test(trainData)
print(confusion)
