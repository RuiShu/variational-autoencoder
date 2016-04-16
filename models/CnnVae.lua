require 'models.Vae'
local CnnVae, parent = torch.class('CnnVae', 'Vae')
local c = require 'trepl.colorize'
require 'nngraph'
require 'nnutils.init'

function CnnVae:build(struct)
   print("CnnVae:build is abstract method. Build it yourself.")
   os.exit()
end

function CnnVae:cuda()
   parent.cuda(self)
   require 'cudnn'
   -- cudnn.benchmark = true
   cudnn.fastest = true
   cudnn.convert(self.encoder, cudnn)
   cudnn.convert(self.decoder, cudnn)
   return self
end
