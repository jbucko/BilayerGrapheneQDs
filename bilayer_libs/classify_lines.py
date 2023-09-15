
def classify(m,UinmeV,VinmeV,tau,arr,BinTmax):
	"""
	this function classifies the lines manually.
	From inspection of density plots, there are 1-3 lines (in case m=2,tau=-1), there is around U,V=(40,40) small region with 0 states.
	the bottommost state is more flat/narrow than other ones. The respective cases should leads to high precision classification of lines.
	"""
	l = len(arr)
	if l==0:
		return [0]
	if m==0 and tau==1:
		if l==1:
			return [2]
		if l==2:
			if abs(arr[0][-1]-arr[0][0])<1.5*BinTmax/2.5:
				return [1,2]
			else:
				return [2,3]
		if l==3:
			return [1,2,3]


	if m==0 and tau==-1:
		if l==1:
			if arr[0][0]<(UinmeV -VinmeV/2 + 0.6*VinmeV):
				return [2]
			else:
				return [3]
		if l==2:
			if abs(arr[0][-1]-arr[0][0])<1.5*BinTmax/2.5:
				return [1,2]
			else:
				return [2,3]
		if l==3:
			return [1,2,3]


	if m==1 and tau==1:
		if l==1:
			return [3]
		if l==2:
			if abs(arr[0][-1]-arr[0][0])<1.5*BinTmax/2.5:
				return [1,2]
			else:
				return [2,3]
		if l==3:
			return [1,2,3]


	if m==1 and tau==-1:
		if l==1:
			return [2]
		if l==2:
			if abs(arr[0][-1]-arr[0][0])<1.5*BinTmax/2.5:
				return [1,2]
			else:
				return [2,3]
		if l==3:
			return [1,2,3]

	if m==2 and tau==1:
		if l==1:
			return [2]
		if l==2:
			if abs(arr[0][-1]-arr[0][0])<1.5*BinTmax/2.5:
				return [1,2]
			else:
				return [2,3]
		if l==3:
			return [1,2,3]

	if m==2 and tau==-1:
		if l==1:
			return [2]
		if l==2:
			if abs(arr[0][-1]-arr[0][0])<1.5*BinTmax/2.5:
				return [1,2]
			else:
				return [2,3]
		if l==3:
			return [1,2,3]

	if m==-1 and tau==1:
		if l==1:
			return [2]
		if l==2:
			if abs(arr[0][-1]-arr[0][0])<1.5*BinTmax/2.5:
				return [1,2]
			else:
				return [2,3]
		if l==3:
			return [1,2,3]


	if m==-1 and tau==-1:
		if l==1:
			return [3]
		if l==2:
			if abs(arr[0][-1]-arr[0][0])<1.5*BinTmax/2.5:
				return [1,2]
			else:
				return [2,3]
		if l==3:
			return [1,2,3]


	if m==-2 and tau==1:
		if l==1:
			return [2]
		if l==2:
			if abs(arr[0][-1]-arr[0][0])<1.5*BinTmax/2.5:
				return [1,2]
			else:
				return [2,3]
		if l==3:
			return [1,2,3]

	if m==-2 and tau==-1:
		if l==1:
			return [2]
		if l==2:
			if abs(arr[0][-1]-arr[0][0])<1.5*BinTmax/2.5:
				return [1,2]
			else:
				return [2,3]
		if l==3:
			return [1,2,3]
