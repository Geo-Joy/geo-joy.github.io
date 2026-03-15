# Patch for Ruby 3.2+ which removed Object#tainted? / Object#taint
# Required for Liquid 4.0.3 compatibility
class Object
  def tainted?
    false
  end

  def taint
    self
  end
end
