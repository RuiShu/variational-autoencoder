require 'nn'

local Zero, parent = torch.class('nn.Zero', 'nn.Module')

function Zero:__init()
   parent.__init(self)
end

function Zero:updateOutput(input)
   self.output:resizeAs(input)
   self.output:zero()
   return self.output
end

function Zero:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()
   return self.gradInput
end
