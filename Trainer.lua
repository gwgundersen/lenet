-- ============================================================================
-- Class for training datasets.
-- ============================================================================

require 'optim'


Trainer = {}
Trainer.__index = Trainer
setmetatable(Trainer, {
    __call = function (cls, ...)
        return cls.new(...)
    end
})


function Trainer.new(model, criterion, classes, batchSize, geometry)
    local self = setmetatable({}, Trainer)
    self.model = model
    self.criterion = criterion
    self.batchSize = batchSize
    self.geometry = geometry
    local parameters, gradParameters = model:getParameters()
    self.parameters = parameters
    self.gradParameters = gradParameters
    self.confusion = optim.ConfusionMatrix(classes)
    return self
end


function Trainer:getInputsAndTargetsTensors(dataset, t)
    local inputs = torch.Tensor(
        self.batchSize,
        self.geometry[1],
        self.geometry[2],
        self.geometry[3]
    )
    local targets = torch.Tensor(self.batchSize)
    local k = 1
    for i = t, math.min(t+self.batchSize-1, dataset:size()) do
        local sample = dataset[i]
        local input = sample[1]:clone()
        -- max(1) returns the argument of the maximum element, i.e. the index
        -- of the non-zero value in a one-hot array.
        local _, target = sample[2]:clone():max(1)
        target = target:squeeze()
        inputs[k] = input
        targets[k] = target
        k = k + 1
    end
    return inputs, targets
end


function Trainer:train(dataset)
    print('Training...')
    for t = 1, dataset:size(), self.batchSize do
        local inputs, targets = self:getInputsAndTargetsTensors(dataset, t)

        -- Optimization functions take closure with access to model and data.
        local function f_eval(x)
            -- Reset gradients.
            self.gradParameters:zero()
            local outputs = self.model:forward(inputs)
            local f       = self.criterion:forward(outputs, targets)
            local df_do   = self.criterion:backward(outputs, targets)
            self.model:backward(inputs, df_do)

            for i = 1, self.batchSize do
                self.confusion:add(outputs[i], targets[i])
            end

            return f, self.gradParameters
        end

        optim.adam(f_eval, self.parameters)
    end
end


function Trainer:test(dataset)
    print('Testing...')
    for t = 1, dataset:size(), self.batchSize do
        -- disp progress
        xlua.progress(t, dataset:size())
        local inputs, targets = self:getInputsAndTargetsTensors(dataset, t)
        local preds = self.model:forward(inputs)

        for i = 1, self.batchSize do
            self.confusion:add(preds[i], targets[i])
        end
    end
    return self.confusion
end
